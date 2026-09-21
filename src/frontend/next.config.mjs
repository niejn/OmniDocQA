/** @type {import('next').NextConfig} */

/*
 * 后端 FastAPI 源。rewrites 在 dev-server 启动 / build 时对 next.config.mjs 求值，
 * 此处 process.env 是启动时进程环境（与 src/app/api/** 代理路由读取的是同一个
 * BACKEND_API_BASE_URL 变量），而非浏览器端 env——不存在构建期固化到客户端的问题。
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
