export default defineNuxtConfig({
  modules: ['@nuxt/ui'],
  css: ['~/assets/css/main.css'],
  devtools: { enabled: true },
  ssr: true,
  app: {
    head: {
      htmlAttrs: { lang: 'ko' },
      title: 'Qwen Local — 기기 안에서 답하는 AI',
      meta: [
        {
          name: 'description',
          content: '학생, 교사, 일반 사용자를 위한 Qwen3.5-4B WebGPU 로컬 AI 채팅'
        }
      ]
    }
  },
  compatibilityDate: '2026-08-10'
})
