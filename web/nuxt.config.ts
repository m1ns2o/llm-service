import { fileURLToPath } from 'node:url'

const androidBuild = process.env.NUXT_ANDROID_BUILD === '1'

export default defineNuxtConfig({
  buildDir: androidBuild ? '.nuxt-android' : '.nuxt',
  alias: androidBuild
    ? { '@mlc-ai/web-llm': fileURLToPath(new URL('./app/stubs/web-llm.ts', import.meta.url)) }
    : {},
  modules: ['@nuxt/ui', '@nuxtjs/mdc'],
  css: ['~/assets/css/main.css'],
  devtools: { enabled: true },
  ssr: false,
  app: {
    baseURL: androidBuild ? '/assets/web/' : '/',
    // Android's aapt ignores asset-directory names beginning with `_`.
    buildAssetsDir: androidBuild ? '/nuxt-assets/' : '/_nuxt/',
    head: {
      htmlAttrs: { lang: 'ko' },
      title: 'Qwen Local — 기기 안에서 답하는 AI',
      meta: [
        {
          name: 'viewport',
          content: 'width=device-width, initial-scale=1, viewport-fit=cover'
        },
        {
          name: 'description',
          content: '학생, 교사, 일반 사용자를 위한 Qwen3.5-2B WebGPU 로컬 AI 채팅'
        }
      ]
    }
  },
  compatibilityDate: '2026-08-10'
})
