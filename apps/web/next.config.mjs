/** @type {import('next').NextConfig} */
const nextConfig = {
  // The browser never talks to FastAPI directly — it hits our BFF route
  // (app/api/chat), which proxies server-side to BACKEND_URL. So there is no
  // CORS surface and no backend URL/secret shipped to the client.
  reactStrictMode: true,
};

export default nextConfig;
