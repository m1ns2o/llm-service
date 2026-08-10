export type LocalModelChoice =
  | 'qwen35-08b'
  | 'qwen35-2b'
  | 'qwen35-4b'
  | 'qwen35-9b'
  | 'lfm2-8b'

export type LocalModelFamily = 'qwen' | 'lfm'

export interface LocalModelDefinition {
  id: string
  family: LocalModelFamily
  label: string
  shortLabel: string
  source: string
  description: string
  storageHint: string
  estimatedDownloadGb: number
  estimatedVramMb: number
  runtime: string
  quality: 'compact' | 'balanced' | 'quality' | 'high'
  speed: 'fast' | 'balanced' | 'quality'
  autoEligible: boolean
  caveat?: string
}

export const LOCAL_MODELS: Record<LocalModelChoice, LocalModelDefinition> = {
  'qwen35-08b': {
    id: 'Qwen3.5-0.8B-q4f16_1-MLC',
    family: 'qwen',
    label: 'Qwen3.5-0.8B',
    shortLabel: '0.8B · 저사양',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC',
    description: '저사양 PC와 작은 브라우저 메모리를 위한 최소 다운로드 모델입니다.',
    storageHint: '약 1.6GB VRAM이 필요하며, 최초 실행 후 브라우저 캐시에서 재사용합니다.',
    estimatedDownloadGb: 0.65,
    estimatedVramMb: 1629.49,
    runtime: 'WebLLM · MLC q4f16_1',
    quality: 'compact',
    speed: 'fast',
    autoEligible: true
  },
  'qwen35-2b': {
    id: 'Qwen3.5-2B-q4f16_1-MLC',
    family: 'qwen',
    label: 'Qwen3.5-2B',
    shortLabel: '2B · 균형',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-2B-q4f16_1-MLC',
    description: '대부분의 내장 그래픽 노트북에서 속도와 품질의 균형을 맞춘 기본 모델입니다.',
    storageHint: '약 2.25GB VRAM이 필요하며, 최초 실행 후 브라우저 캐시에서 재사용합니다.',
    estimatedDownloadGb: 1.25,
    estimatedVramMb: 2245.44,
    runtime: 'WebLLM · MLC q4f16_1',
    quality: 'balanced',
    speed: 'fast',
    autoEligible: true
  },
  'qwen35-4b': {
    id: 'Qwen3.5-4B-q4f16_1-MLC',
    family: 'qwen',
    label: 'Qwen3.5-4B',
    shortLabel: '4B · 품질',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC',
    description: '여유 있는 WebGPU에서 범용 답변 품질을 높이는 권장 모델입니다.',
    storageHint: '약 3.87GB VRAM이 필요하며, 최초 실행 후 브라우저 캐시에서 재사용합니다.',
    estimatedDownloadGb: 2.55,
    estimatedVramMb: 3867.82,
    runtime: 'WebLLM · MLC q4f16_1',
    quality: 'quality',
    speed: 'balanced',
    autoEligible: true
  },
  'qwen35-9b': {
    id: 'Qwen3.5-9B-q4f16_1-MLC',
    family: 'qwen',
    label: 'Qwen3.5-9B',
    shortLabel: '9B · 고품질',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-9B-q4f16_1-MLC',
    description: '고사양 외장 GPU에서 범용 품질을 우선하는 선택 모델입니다.',
    storageHint: '약 6.43GB VRAM이 필요합니다. 브라우저가 실제 VRAM을 공개하지 않아 자동 다운로드하지 않습니다.',
    estimatedDownloadGb: 5.15,
    estimatedVramMb: 6433.01,
    runtime: 'WebLLM · MLC q4f16_1',
    quality: 'high',
    speed: 'quality',
    autoEligible: false
  },
  'lfm2-8b': {
    id: 'LiquidAI/LFM2-8B-A1B-ONNX',
    family: 'lfm',
    label: 'LFM2-8B-A1B',
    shortLabel: '8B MoE · 고속',
    source: 'https://huggingface.co/LiquidAI/LFM2-8B-A1B-ONNX',
    description: '텍스트 생성 속도를 우선하는 8B MoE 모델입니다. 선택할 때만 내려받습니다.',
    storageHint: 'q4f16 ONNX 파일 약 4.78GB를 받습니다. 한국어 언어 준수는 Qwen보다 불안정할 수 있습니다.',
    estimatedDownloadGb: 4.78,
    estimatedVramMb: 5200,
    runtime: 'Transformers.js · WebGPU QMoE q4f16',
    quality: 'quality',
    speed: 'fast',
    autoEligible: false,
    caveat: '이전 실측에서 빠르지만 일부 한국어 질문에 영어로 답한 사례가 있어 속도 모드로 분리했습니다.'
  }
}

export const LOCAL_MODEL_CHOICES = Object.keys(LOCAL_MODELS) as LocalModelChoice[]

export function isLocalModelChoice(value: unknown): value is LocalModelChoice {
  return typeof value === 'string' && value in LOCAL_MODELS
}
