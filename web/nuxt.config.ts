export default defineNuxtConfig({
  modules: ['@nuxt/ui'],
  css: ['~/assets/css/main.css'],
  devtools: { enabled: true },
  ssr: true,
  routeRules: {
    '/**': {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'same-origin'
      }
    }
  },
  app: {
    head: {
      htmlAttrs: { lang: 'ko' },
      title: 'Local AI — 기기 안에서 답하는 AI',
      meta: [
        {
          name: 'description',
          content: 'Qwen3.5-4B와 LFM2.5-8B를 선택할 수 있는 WebGPU 로컬 AI 채팅'
        }
      ]
    }
  },
  compatibilityDate: '2026-08-10'
})
