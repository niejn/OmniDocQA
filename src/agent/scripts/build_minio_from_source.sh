#!/usr/bin/env bash
# Build the MinIO community image FROM SOURCE for tags that ship no prebuilt
# docker image (2025-09-07 / 2025-10-15 and anything later: their GitHub release
# bodies say "clone the source and build"; docs/DEV_PROGRESS.md 09-19/20).
#
# How upstream builds (verified against the tagged repo, NOT guessed):
#   Makefile `build`: CGO_ENABLED=0 go build -tags kqueue -trimpath
#                     --ldflags "$(go run buildscripts/gen-ldflags.go $VERSION)"
#                     -o ./minio                     ← linux binary in context
#   Dockerfile:       FROM minio/minio:latest; COPY ./minio /usr/bin/minio ...
# So the automated flow is:
#   1. shallow clone the tag
#   2. compile the linux binary INSIDE a golang container (host needs no Go):
#      GOPROXY=goproxy.cn (proxy.golang.org unreachable from this network),
#      ldflags replicated from buildscripts/gen-ldflags.go (git on the host
#      supplies commit ids; container has no git)
#   3. satisfy `FROM minio/minio:latest` offline by retagging our known-good
#      RELEASE.2025-07-23T15-54-02Z image (base is only chmod+COPY host layout)
#   4. docker build → verify `--version` reports the tag
#
# Usage:
#   bash scripts/build_minio_from_source.sh [TAG] [--switch]
#     TAG      default RELEASE.2025-10-15T17-29-55Z (final community release,
#              includes the STS session-policy CVE fix)
#     --switch update docker-compose.rag.yml and recreate rag-minio
# Exit codes: 0 ok | 1 build/verify failed | 2 tag/repo error.
set -euo pipefail

TAG="${1:-RELEASE.2025-10-15T17-29-55Z}"
SWITCH="false"
[ "${2:-}" = "--switch" ] && SWITCH="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC_DIR="$REPO_ROOT/.minio_src"
COMPOSE="$REPO_ROOT/docker-compose.rag.yml"
GOPROXY_URL="https://goproxy.cn,direct"
KNOWN_GOOD_BASE="minio/minio:RELEASE.2025-07-23T15-54-02Z"

log() { echo "[MinIO-Build] $*"; }

# ── 1. source: shallow clone / fetch the tag ─────────────────────────────
if [ -d "$SRC_DIR/.git" ]; then
  log "fetching tag $TAG into existing clone..."
  git -C "$SRC_DIR" fetch --depth 1 --force origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$SRC_DIR" checkout -f "$TAG"
else
  log "shallow-cloning minio at $TAG (first run downloads a few hundred MB)..."
  git -c core.autocrlf=false clone --depth 1 --branch "$TAG" https://github.com/minio/minio.git "$SRC_DIR"
fi
git -C "$SRC_DIR" config core.autocrlf false
git -C "$SRC_DIR" checkout -f "$TAG" >/dev/null 2>&1 || true
find "$SRC_DIR" -name "*.sh" -exec sed -i "s/$//" {} + 2>/dev/null || true
git -C "$SRC_DIR" status >/dev/null
[ -f "$SRC_DIR/Dockerfile" ] || { log "ERROR: no Dockerfile at repo root"; exit 2; }
[ -f "$SRC_DIR/go.mod" ] || { log "ERROR: no go.mod"; exit 2; }
log "source ready: $(git -C "$SRC_DIR" rev-parse --short HEAD)"

# ── 2. compile the linux binary inside a golang container ────────────────
# go version from go.mod ("go 1.24.8" → golang:1.24-alpine)
GO_MINOR="$(grep -E '^go [0-9]+\.[0-9]+' "$SRC_DIR/go.mod" | awk '{print $2}' | cut -d. -f1,2)"
GO_IMAGE="golang:${GO_MINOR}-alpine"
log "toolchain: $GO_IMAGE (from go.mod)"
if ! docker image inspect "$GO_IMAGE" >/dev/null 2>&1; then
  log "pre-pulling toolchain via DDN..."
  bash "$SCRIPT_DIR/ddn_pull.sh" "$GO_IMAGE"
fi

COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"
SHORT="$(git -C "$SRC_DIR" rev-parse --short=12 HEAD)"
COPYRIGHT_YEAR="$(date +%Y)"
# replicate buildscripts/gen-ldflags.go output (host git, container GOPATH/GOROOT)
LDFLAGS="-s -w"
LDFLAGS+=" -X github.com/minio/minio/cmd.Version=$TAG"
LDFLAGS+=" -X github.com/minio/minio/cmd.CopyrightYear=$COPYRIGHT_YEAR"
LDFLAGS+=" -X github.com/minio/minio/cmd.ReleaseTag=$TAG"
LDFLAGS+=" -X github.com/minio/minio/cmd.CommitID=$COMMIT"
LDFLAGS+=" -X github.com/minio/minio/cmd.ShortCommitID=$SHORT"
LDFLAGS+=" -X github.com/minio/minio/cmd.GOPATH=/go"
LDFLAGS+=" -X github.com/minio/minio/cmd.GOROOT=/usr/local/go"

log "compiling linux/amd64 binary in container (go module downloads take a few minutes)..."
# MSYS_NO_PATHCONV: Git Bash would otherwise rewrite -w /work into a Windows path
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$SRC_DIR":/work -w /work \
  -e "GOPROXY=$GOPROXY_URL" \
  -e CGO_ENABLED=0 -e GOOS=linux -e GOARCH=amd64 \
  -e GOPATH=/go -e GOROOT=/usr/local/go \
  "$GO_IMAGE" \
  go build -tags kqueue -trimpath --ldflags "$LDFLAGS" -o /work/minio .
[ -f "$SRC_DIR/minio" ] || { log "ERROR: go build produced no binary"; exit 1; }
log "binary built: $(du -h "$SRC_DIR/minio" | cut -f1)"

# ── 3. satisfy `FROM minio/minio:latest` offline ─────────────────────────
if ! docker image inspect minio/minio:latest >/dev/null 2>&1; then
  if docker image inspect "$KNOWN_GOOD_BASE" >/dev/null 2>&1; then
    docker tag "$KNOWN_GOOD_BASE" minio/minio:latest
    log "base minio/minio:latest <- retagged $KNOWN_GOOD_BASE (layout host only)"
  else
    log "pre-pulling base via DDN..."
    bash "$SCRIPT_DIR/ddn_pull.sh" minio/minio:latest
  fi
fi

# ── 4. docker build (COPY binary into base) + verify ─────────────────────
docker build -t "minio/minio:$TAG" "$SRC_DIR"
log "image built: minio/minio:$TAG"

VERSION_OUT="$(MSYS_NO_PATHCONV=1 docker run --rm "minio/minio:$TAG" --version 2>&1 \
  || MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /usr/bin/minio "minio/minio:$TAG" --version 2>&1)"
echo "$VERSION_OUT"
echo "$VERSION_OUT" | grep -q "$TAG" || { log "ERROR: built binary does not report $TAG"; exit 1; }
log "version verified ✓"

# ── 5. optional cutover of rag-minio ─────────────────────────────────────
if [ "$SWITCH" = "true" ]; then
  python - "$TAG" "$COMPOSE" <<'PYEOF'
import io, sys, re
tag, path = sys.argv[1], sys.argv[2]
src = io.open(path, encoding="utf-8").read()
new_src, n = re.subn(r'(  rag-minio:\n(?:#[^\n]*\n)*\s*image: minio/minio:)[A-Za-z0-9.T-]+',
                     r'\g<1>' + tag, src)
assert n == 1, f"expected exactly one rag-minio image line, patched {n}"
io.open(path, "w", encoding="utf-8", newline="\n").write(new_src)
print("[MinIO-Build] compose image tag ->", tag)
PYEOF
  docker compose -f "$COMPOSE" up -d rag-minio
  log "waiting for rag-minio healthy..."
  STATE="unknown"
  for _ in $(seq 1 30); do
    STATE="$(docker inspect -f '{{.State.Health.Status}}' rag-minio 2>/dev/null || echo starting)"
    [ "$STATE" = "healthy" ] && break
    sleep 3
  done
  log "rag-minio state: $STATE (bucket volume untouched)"
  log "switch complete — verify assets with the /documents page-image endpoint"
else
  log "built and tagged locally. to switch rag-minio onto it:"
  log "  bash scripts/build_minio_from_source.sh $TAG --switch"
fi
