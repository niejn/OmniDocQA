#!/usr/bin/env bash
# docker pull 加速脚本：Docker Hub 镜像经华为云 DDN 公共代理拉取后 retag 回官方名。
#
# 背景（2026-09-19 实测）：本机 daemon 配置的加速器（docker.xuanyuan.me / docker.m.daocloud.io）
# 对部分镜像（如 minio/minio）回源 403；DDN 源可正常 plain pull，但它是"仓库名内嵌代理"
# （ddn-k8s/docker.io 是 SWR 仓库名空间），无法写进 daemon.json 的 registry-mirrors，
# 因此用本脚本固化显式前缀用法。
#
# 用法:
#   ./scripts/ddn_pull.sh minio/minio:RELEASE.2025-07-23T15-54-02Z
#   ./scripts/ddn_pull.sh bitnami/redis:latest          # 任意 docker.io 镜像
#   ./scripts/ddn_pull.sh minio/minio:RELEASE.2025-06-13T11-33-47Z --no-rm   # 保留 DDN 前缀 tag
set -euo pipefail

DDN_BASE="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io"

[ $# -ge 1 ] || { echo "usage: $0 <docker.io-repo:tag> [--no-rm]"; exit 1; }
IMAGE="$1"; KEEP="${2:-}"

docker pull "${DDN_BASE}/${IMAGE}"
if [ "$KEEP" != "--no-rm" ]; then
  docker tag "${DDN_BASE}/${IMAGE}" "$IMAGE"
  docker rmi "${DDN_BASE}/${IMAGE}" >/dev/null
  echo ">> $IMAGE (via DDN proxy, official tag restored)"
fi
