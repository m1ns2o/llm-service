import type { LocalModelChoice } from './localModelCatalog'

export interface BrowserHardwareProfile {
  webgpuAvailable: boolean
  adapterLabel: string
  vendor: string
  architecture: string
  logicalCores: number
  deviceMemoryGb: number | null
  maxBufferSizeMb: number
  maxStorageBufferMb: number
  shaderF16: boolean
  storageUsageGb: number | null
  storageQuotaGb: number | null
  recommendedModel: LocalModelChoice
  recommendationReason: string
  confidence: 'low' | 'medium'
}

const GIB = 1024 ** 3

function normalized(value: unknown) {
  return typeof value === 'string' ? value.trim().toLowerCase() : ''
}

export function recommendModel(profile: Omit<BrowserHardwareProfile, 'recommendedModel' | 'recommendationReason' | 'confidence'>) {
  if (!profile.webgpuAvailable) {
    return {
      model: 'qwen35-08b' as const,
      reason: 'WebGPU를 찾지 못해 가장 작은 모델을 표시했습니다. 이 브라우저에서는 실행되지 않을 수 있습니다.',
      confidence: 'low' as const
    }
  }

  const lowMemory = profile.deviceMemoryGb !== null && profile.deviceMemoryGb <= 4
  const lowCompute = profile.logicalCores > 0 && profile.logicalCores <= 4
  const smallBuffer = profile.maxBufferSizeMb > 0 && profile.maxBufferSizeMb < 1024
  if (smallBuffer || !profile.shaderF16 || (lowMemory && lowCompute)) {
    return {
      model: 'qwen35-08b' as const,
      reason: '메모리·CPU·WebGPU 기능을 보수적으로 판정해 다운로드가 작은 0.8B를 선택했습니다.',
      confidence: 'medium' as const
    }
  }

  const capableVendor = ['amd', 'nvidia', 'apple'].some(name => profile.vendor.includes(name))
  const largeBuffers = profile.maxBufferSizeMb >= 1536 && profile.maxStorageBufferMb >= 512
  const enoughSystemMemory = profile.deviceMemoryGb !== null && profile.deviceMemoryGb >= 8
  if (capableVendor && largeBuffers && enoughSystemMemory && profile.logicalCores >= 8) {
    return {
      model: 'qwen35-4b' as const,
      reason: 'WebGPU 버퍼 한도와 시스템 자원이 충분해 품질 우선 4B를 추천합니다.',
      confidence: 'medium' as const
    }
  }

  return {
    model: 'qwen35-2b' as const,
    reason: '브라우저가 실제 VRAM을 공개하지 않아 호환성이 높은 2B를 안전한 기본값으로 선택했습니다.',
    confidence: 'low' as const
  }
}

export async function detectBrowserHardware(): Promise<BrowserHardwareProfile> {
  const extendedNavigator = navigator as Navigator & { deviceMemory?: number }
  const adapter = navigator.gpu
    ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    : null
  const info = adapter?.info
  const storage = navigator.storage?.estimate ? await navigator.storage.estimate() : {}
  const base = {
    webgpuAvailable: Boolean(adapter),
    adapterLabel: info ? [info.vendor, info.architecture].filter(Boolean).join(' · ') : '',
    vendor: normalized(info?.vendor),
    architecture: normalized(info?.architecture),
    logicalCores: navigator.hardwareConcurrency || 0,
    deviceMemoryGb: typeof extendedNavigator.deviceMemory === 'number'
      ? extendedNavigator.deviceMemory
      : null,
    maxBufferSizeMb: adapter ? Math.round(Number(adapter.limits.maxBufferSize) / 1024 ** 2) : 0,
    maxStorageBufferMb: adapter
      ? Math.round(Number(adapter.limits.maxStorageBufferBindingSize) / 1024 ** 2)
      : 0,
    shaderF16: adapter?.features.has('shader-f16') || false,
    storageUsageGb: typeof storage.usage === 'number' ? storage.usage / GIB : null,
    storageQuotaGb: typeof storage.quota === 'number' ? storage.quota / GIB : null
  }
  const recommendation = recommendModel(base)
  return {
    ...base,
    recommendedModel: recommendation.model,
    recommendationReason: recommendation.reason,
    confidence: recommendation.confidence
  }
}

export function useHardwareProfile() {
  const profile = useState<BrowserHardwareProfile | null>('browser-hardware-profile', () => null)
  const profiling = useState<boolean>('browser-hardware-profiling', () => false)
  const profileError = useState<string | null>('browser-hardware-error', () => null)

  async function detect() {
    if (!import.meta.client) return null
    profiling.value = true
    profileError.value = null
    try {
      profile.value = await detectBrowserHardware()
      return profile.value
    } catch (error: unknown) {
      profileError.value = error instanceof Error ? error.message : String(error)
      return null
    } finally {
      profiling.value = false
    }
  }

  return { profile, profiling, profileError, detect }
}
