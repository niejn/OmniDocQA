/** @type {import('next').NextConfig} */

/*
 * 后端 FastAPI 源。⚠️ rewrites 的 destination 在 `next build` 时求值并固化进
 * .next/routes-manifest.json（`next start` 不会重新求值）——部署时必须在
 * 构建命令上带上 BACKEND_API_BASE_URL（与 src/app/api/** 代理路由的运行时
 * env 读取不同，那 20 个 route 是真运行时读 process.env）。
 */
const BACKEND_API_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        /*
         * Bug fix：后端 document_service.py 生成的图文证据 image_url 是相对路径
         * /agent/api/documents/page-image?...，此前前端源（:3000）上没有 /agent
         * 路由/rewrite，<img src> 必然 404。该 rewrite 让 /agent/** 整体回源后端。
         *
         * 安全性考量：这条 rewrite 把后端 /agent 下「全部」端点都暴露到前端源——
         * 此前经 :3000 只能触达 src/app/api/** 显式代理的 ~20 个端点，现在任何能
         * 访问 :3000 的客户端都能以同等权限访问后端其余端点（入库、评测、报告等）。
         * 后端自身无鉴权，因此这不产生新的越权路径，但扩大了暴露面：
         *   - 本地 / 内网部署（前后端同一信任边界）：可接受，正是本项目目标形态；
         *   - 公网部署：应给后端加鉴权，或把 source 收窄为
         *     "/agent/api/documents/page-image" 一类白名单前缀。
         * 优先级说明：Next 将数组形式 rewrites（afterFiles）排在文件系统路由之后，
         * src/app/api/** 代理路由不受影响，仅未匹配任何路由的 /agent/** 走回源。
         */
        source: "/agent/:path*",
        destination: `${BACKEND_API_BASE_URL}/agent/:path*`
      }
    ];
  }
};

export default nextConfig;
