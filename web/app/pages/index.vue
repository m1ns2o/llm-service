<script setup lang="ts">
import {
  LOCAL_MODELS,
  LOCAL_MODEL_CHOICES,
  isLocalModelChoice,
  type LocalModelChoice
} from '~/composables/localModelCatalog'
import type { QwenModelChoice } from '~/composables/useLocalQwen'

type ChatStatus = 'ready' | 'submitted' | 'streaming' | 'error'
type ModelChoice = LocalModelChoice

interface TextPart {
  type: 'text'
  text: string
}

interface LocalMessage {
  id: string
  role: 'user' | 'assistant'
  parts: TextPart[]
  model?: ModelChoice
  reasoning?: string
  createdAt: string
  metrics?: {
    decodeTokensPerSecond?: number
    timeToFirstTokenSeconds?: number
    timeToFirstVisibleTokenSeconds?: number
  }
}

interface Conversation {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: LocalMessage[]
}

const STORAGE_KEY = 'local-ai-chat-v2'
const SYSTEM_PROMPT = '당신은 학생, 교사, 일반 사용자를 돕는 친절하고 정확한 한국어 AI입니다. 결론부터 간결하게 답하세요. 사용자가 지정한 단위와 조건을 그대로 유지하고 계산은 한 번 검산하세요. 학습 설명에는 이해 수준에 맞는 예시를 사용하되, 필요하지 않은 확인 질문은 덧붙이지 마세요. 모르는 내용은 추측하지 말고 불확실성을 분명히 밝히세요.'
const MAX_CONTEXT_MESSAGES = 6
const MAX_CONTEXT_CHARACTERS = 1200
const MAX_OUTPUT_TOKENS = 160
const STREAM_RENDER_INTERVAL_MS = 50

const toast = useToast()
const colorMode = useColorMode()
const prompt = ref('')
const conversations = ref<Conversation[]>([])
const activeId = ref('')
const chatStatus = ref<ChatStatus>('ready')
const lastError = ref<string | null>(null)
const copiedMessageId = ref<string | null>(null)
const selectedModel = ref<ModelChoice>('qwen35-2b')
const modelManagerOpen = ref(false)
const cacheStates = ref<Record<ModelChoice, boolean>>({
  'qwen35-08b': false,
  'qwen35-2b': false,
  'qwen35-4b': false,
  'qwen35-9b': false,
  'lfm2-8b': false
})
const cacheChecking = ref(false)
const cacheActionModel = ref<ModelChoice | null>(null)
const initialSelectionComplete = ref(false)
const operationElapsedSeconds = ref(0)
let operationTimer: ReturnType<typeof setInterval> | null = null
const modelOptions = LOCAL_MODEL_CHOICES.map(value => ({
  value,
  label: `${LOCAL_MODELS[value].label} · ${LOCAL_MODELS[value].shortLabel}`
}))

const {
  modelState: qwenModelState,
  modelProgress: qwenModelProgress,
  modelStatusText: qwenModelStatusText,
  modelError: qwenModelError,
  webgpuAvailable,
  webgpuAdapterLabel,
  activeModel,
  loadModel: loadQwenModel,
  unloadModel: unloadQwenModel,
  beginGeneration: beginQwenGeneration,
  wasInterrupted: wasQwenInterrupted,
  stopGeneration: stopQwenGeneration,
  isModelCached: isQwenModelCached,
  deleteCachedModel: deleteQwenCachedModel
} = useLocalQwen()

const {
  modelState: lfmModelState,
  modelProgress: lfmModelProgress,
  modelStatusText: lfmModelStatusText,
  modelError: lfmModelError,
  active: lfmActive,
  loadModel: loadLfmModel,
  generate: generateLfm,
  stopGeneration: stopLfmGeneration,
  unloadModel: unloadLfmModel,
  isModelCached: isLfmModelCached,
  deleteCachedModel: deleteLfmCachedModel
} = useLocalLfmWebGpu()

const { profile, profiling, profileError, detect: detectHardware } = useHardwareProfile()

const activeConversation = computed(() => conversations.value.find(item => item.id === activeId.value))
const messages = computed(() => activeConversation.value?.messages || [])
const conversationSummaries = computed(() => [...conversations.value]
  .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  .map(({ id, title, updatedAt }) => ({ id, title, updatedAt })))
const selectedIsLfm = computed(() => LOCAL_MODELS[selectedModel.value].family === 'lfm')
const modelState = computed(() => selectedIsLfm.value ? lfmModelState.value : qwenModelState.value)
const modelProgress = computed(() => selectedIsLfm.value ? lfmModelProgress.value : qwenModelProgress.value)
const modelStatusText = computed(() => selectedIsLfm.value ? lfmModelStatusText.value : qwenModelStatusText.value)
const modelError = computed(() => selectedIsLfm.value ? lfmModelError.value : qwenModelError.value)
const isBusy = computed(() => chatStatus.value === 'submitted' || chatStatus.value === 'streaming' || modelState.value === 'loading')
const modelReady = computed(() => selectedIsLfm.value
  ? lfmModelState.value === 'ready' && lfmActive.value
  : qwenModelState.value === 'ready' && activeModel.value === selectedModel.value)
const modelPercent = computed(() => Math.round(modelProgress.value * 100))
const modelDefinition = computed(() => LOCAL_MODELS[selectedModel.value])
const modelLabel = computed(() => modelDefinition.value.label)
const modelDescription = computed(() => modelDefinition.value.description)
const modelStorageHint = computed(() => modelDefinition.value.storageHint)
const runtimeLabel = computed(() => [
  'WebGPU',
  profile.value?.adapterLabel || webgpuAdapterLabel.value,
  modelDefinition.value.runtime
].filter(Boolean).join(' · '))
const recommendedModelLabel = computed(() => profile.value
  ? LOCAL_MODELS[profile.value.recommendedModel].label
  : '')
const cachedCount = computed(() => LOCAL_MODEL_CHOICES.filter(model => cacheStates.value[model]).length)
const hardwareSummary = computed(() => {
  if (profiling.value) return 'PC 사양 확인 중'
  if (!profile.value) return profileError.value || 'PC 사양 확인 전'
  const memory = profile.value.deviceMemoryGb ? `${profile.value.deviceMemoryGb}GB 메모리 힌트` : '메모리 미공개'
  const adapter = profile.value.adapterLabel || 'GPU 정보 미공개'
  return `${adapter} · ${profile.value.logicalCores} threads · ${memory} · 최대 버퍼 ${profile.value.maxBufferSizeMb}MB`
})
const operationStatusText = computed(() => {
  if (modelState.value === 'loading') return `모델 로딩·WebGPU 컴파일 중 · ${operationElapsedSeconds.value}초`
  if (chatStatus.value === 'submitted') return `요청 준비 중 · ${operationElapsedSeconds.value}초`
  if (chatStatus.value === 'streaming') return `답변 스트리밍 중 · ${operationElapsedSeconds.value}초`
  return ''
})
const modelBadgeColor = computed(() => {
  if (modelState.value === 'ready') return 'success'
  if (modelState.value === 'error') return 'error'
  if (modelState.value === 'loading' || modelState.value === 'checking') return 'warning'
  return 'neutral'
})
const starterPrompts = [
  { icon: 'i-lucide-graduation-cap', label: '개념을 쉽게 설명해줘' },
  { icon: 'i-lucide-notebook-tabs', label: '수업 활동을 만들어줘' },
  { icon: 'i-lucide-file-text', label: '글의 핵심을 요약해줘' },
  { icon: 'i-lucide-code-2', label: '코드를 예제로 알려줘' }
]

function makeId() {
  return crypto.randomUUID()
}

function newConversation() {
  const now = new Date().toISOString()
  const conversation: Conversation = {
    id: makeId(),
    title: '새 대화',
    createdAt: now,
    updatedAt: now,
    messages: []
  }
  conversations.value.unshift(conversation)
  activeId.value = conversation.id
  prompt.value = ''
  lastError.value = null
  chatStatus.value = 'ready'
  persist()
}

function selectConversation(id: string) {
  activeId.value = id
}

function removeConversation(id: string) {
  conversations.value = conversations.value.filter(item => item.id !== id)
  if (activeId.value === id) {
    if (conversations.value[0]) activeId.value = conversations.value[0].id
    else newConversation()
  }
  persist()
}

function persist() {
  if (!import.meta.client) return
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    activeId: activeId.value,
    conversations: conversations.value,
    selectedModel: selectedModel.value
  }))
}

function stopOperationTimer() {
  if (operationTimer !== null) clearInterval(operationTimer)
  operationTimer = null
}

function startOperationTimer() {
  stopOperationTimer()
  operationElapsedSeconds.value = 0
  const startedAt = performance.now()
  operationTimer = setInterval(() => {
    operationElapsedSeconds.value = Math.floor((performance.now() - startedAt) / 1000)
  }, 250)
}

function parseOutput(raw: string) {
  const open = raw.indexOf('<think>')
  if (open < 0) return { reasoning: '', answer: raw }

  const close = raw.indexOf('</think>', open + 7)
  if (close < 0) return { reasoning: raw.slice(open + 7).trim(), answer: '' }

  return {
    reasoning: raw.slice(open + 7, close).trim(),
    answer: raw.slice(close + 8).trimStart()
  }
}

function textOf(message: LocalMessage) {
  return message.parts.filter(part => part.type === 'text').map(part => part.text).join('')
}

function recentRequestMessages(conversation: Conversation, excludedId: string) {
  const selected: Array<{ role: LocalMessage['role'], content: string }> = []
  let remainingCharacters = MAX_CONTEXT_CHARACTERS

  for (const message of conversation.messages.filter(item => item.id !== excludedId).slice(-MAX_CONTEXT_MESSAGES).reverse()) {
    if (remainingCharacters <= 0) break
    const content = textOf(message).trim()
    if (!content) continue

    const limitedContent = content.length > remainingCharacters
      ? content.slice(0, remainingCharacters)
      : content
    selected.push({ role: message.role, content: limitedContent })
    remainingCharacters -= limitedContent.length
  }

  return selected.reverse()
}

async function prepareModel(notify = true) {
  if (!modelReady.value) startOperationTimer()
  try {
    if (selectedIsLfm.value) {
      await unloadQwenModel()
      await loadLfmModel()
    } else {
      await unloadLfmModel()
      await loadQwenModel(selectedModel.value as QwenModelChoice)
    }
    cacheStates.value[selectedModel.value] = true
  } catch {
    if (notify) {
      toast.add({ title: '모델 준비 실패', description: modelError.value || '모델을 불러오지 못했습니다.', color: 'error' })
    }
  } finally {
    stopOperationTimer()
  }
}

async function handleModelChange() {
  lastError.value = null
  chatStatus.value = 'ready'
  persist()
  await prepareModel(false)
}

async function refreshCacheStates() {
  if (!import.meta.client || cacheChecking.value) return
  cacheChecking.value = true
  try {
    const qwenModels = LOCAL_MODEL_CHOICES.filter(
      model => LOCAL_MODELS[model].family === 'qwen'
    ) as QwenModelChoice[]
    const qwenStates = await Promise.all(qwenModels.map(model => isQwenModelCached(model)))
    qwenModels.forEach((model, index) => {
      cacheStates.value[model] = Boolean(qwenStates[index])
    })
    cacheStates.value['lfm2-8b'] = await isLfmModelCached()
  } finally {
    cacheChecking.value = false
  }
}

async function activateManagedModel(model: ModelChoice) {
  if (isBusy.value) return
  selectedModel.value = model
  modelManagerOpen.value = false
  persist()
  await prepareModel()
}

async function removeCachedModel(model: ModelChoice) {
  if (isBusy.value) return
  cacheActionModel.value = model
  try {
    if (LOCAL_MODELS[model].family === 'lfm') {
      await deleteLfmCachedModel()
    } else {
      await deleteQwenCachedModel(model as QwenModelChoice)
    }
    cacheStates.value[model] = false
    toast.add({ title: '캐시 삭제 완료', description: `${LOCAL_MODELS[model].label} 모델 파일을 브라우저에서 삭제했습니다.` })
  } catch (error: unknown) {
    toast.add({
      title: '캐시 삭제 실패',
      description: error instanceof Error ? error.message : String(error),
      color: 'error'
    })
  } finally {
    cacheActionModel.value = null
  }
}

async function submitPrompt(value = prompt.value) {
  const content = value.trim()
  if (!content || isBusy.value || !modelReady.value || !activeConversation.value) return

  lastError.value = null
  const conversation = activeConversation.value
  const userMessage: LocalMessage = {
    id: makeId(),
    role: 'user',
    parts: [{ type: 'text', text: content }],
    createdAt: new Date().toISOString()
  }
  conversation.messages.push(userMessage)
  if (conversation.messages.length === 1) conversation.title = content.slice(0, 32)
  conversation.updatedAt = new Date().toISOString()
  prompt.value = ''
  chatStatus.value = 'submitted'
  startOperationTimer()
  persist()

  let assistant: LocalMessage = {
    id: makeId(),
    role: 'assistant',
    parts: [{ type: 'text', text: '' }],
    model: selectedModel.value,
    reasoning: '',
    createdAt: new Date().toISOString()
  }
  conversation.messages.push(assistant)
  const assistantIndex = conversation.messages.length - 1

  function replaceAssistant(changes: Partial<LocalMessage>) {
    assistant = { ...assistant, ...changes }
    conversation.messages.splice(assistantIndex, 1, assistant)
  }

  try {
    const requestMessages = [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      ...recentRequestMessages(conversation, assistant.id)
    ]

    chatStatus.value = 'streaming'
    const generationStartedAt = performance.now()
    let raw = ''
    let lastRenderedAt = 0
    let timeToFirstVisibleTokenSeconds: number | undefined
    let interrupted = false

    function renderAssistantOutput() {
      const parsed = parseOutput(raw)
      replaceAssistant({
        reasoning: parsed.reasoning,
        parts: [{ type: 'text', text: parsed.answer }]
      })
      if (timeToFirstVisibleTokenSeconds === undefined && (parsed.answer || parsed.reasoning)) {
        timeToFirstVisibleTokenSeconds = (performance.now() - generationStartedAt) / 1000
      }
      lastRenderedAt = performance.now()
    }

    if (selectedIsLfm.value) {
      const result = await generateLfm(
        requestMessages,
        MAX_OUTPUT_TOKENS,
        (delta) => {
          raw += delta
          if (delta && performance.now() - lastRenderedAt >= STREAM_RENDER_INTERVAL_MS) {
            renderAssistantOutput()
          }
        }
      )
      interrupted = result.interrupted
      renderAssistantOutput()
      replaceAssistant({
        metrics: {
          decodeTokensPerSecond: result.decodeTokensPerSecond,
          timeToFirstTokenSeconds: result.firstTokenMs === null ? undefined : result.firstTokenMs / 1000,
          timeToFirstVisibleTokenSeconds
        }
      })
    } else {
      const engine = await loadQwenModel(selectedModel.value as QwenModelChoice)
      beginQwenGeneration()
      const chunks = await engine.chat.completions.create({
        messages: requestMessages,
        temperature: 0.2,
        max_tokens: MAX_OUTPUT_TOKENS,
        extra_body: { enable_thinking: false },
        stream: true,
        stream_options: { include_usage: true }
      })
      let usageExtra: { decode_tokens_per_s?: number, time_to_first_token_s?: number } | undefined

      for await (const chunk of chunks) {
        if (wasQwenInterrupted()) break
        const delta = chunk.choices[0]?.delta.content || ''
        raw += delta
        if (chunk.usage?.extra) usageExtra = chunk.usage.extra
        if (delta && performance.now() - lastRenderedAt >= STREAM_RENDER_INTERVAL_MS) {
          renderAssistantOutput()
          await nextTick()
        }
      }
      interrupted = wasQwenInterrupted()
      renderAssistantOutput()
      if (usageExtra || timeToFirstVisibleTokenSeconds !== undefined) {
        replaceAssistant({
          metrics: {
            decodeTokensPerSecond: usageExtra?.decode_tokens_per_s,
            timeToFirstTokenSeconds: usageExtra?.time_to_first_token_s,
            timeToFirstVisibleTokenSeconds
          }
        })
      }
    }
    await nextTick()
    if (!textOf(assistant).trim() && !interrupted) {
      replaceAssistant({ parts: [{ type: 'text', text: '답변을 완성하기 전에 출력 한도에 도달했습니다. 질문을 더 짧게 나누어 다시 시도해 주세요.' }] })
    }
    chatStatus.value = 'ready'
  } catch (error: unknown) {
    lastError.value = error instanceof Error ? error.message : String(error)
    replaceAssistant({ parts: [{ type: 'text', text: '응답을 생성하지 못했습니다. 모델 상태를 확인한 뒤 다시 시도해 주세요.' }] })
    chatStatus.value = 'error'
  } finally {
    stopOperationTimer()
    conversation.updatedAt = new Date().toISOString()
    persist()
  }
}

async function stop() {
  if (selectedIsLfm.value) await stopLfmGeneration()
  else await stopQwenGeneration()
  stopOperationTimer()
  chatStatus.value = 'ready'
  persist()
}

async function copyMessage(message: LocalMessage) {
  await navigator.clipboard.writeText(textOf(message))
  copiedMessageId.value = message.id
  window.setTimeout(() => {
    if (copiedMessageId.value === message.id) copiedMessageId.value = null
  }, 2500)
}

function messageById(id: string) {
  return messages.value.find(message => message.id === id)
}

function textPartsById(id: string) {
  return messageById(id)?.parts || []
}

function textById(id: string) {
  const message = messageById(id)
  return message ? textOf(message) : ''
}

function modelLabelByMessageId(id: string) {
  const model = messageById(id)?.model
  return model && isLocalModelChoice(model) ? LOCAL_MODELS[model].label : ''
}

async function copyMessageById(id: string) {
  const message = messageById(id)
  if (message) await copyMessage(message)
}

function retryLast() {
  const lastUser = [...messages.value].reverse().find(message => message.role === 'user')
  if (!lastUser) return
  const retryText = textOf(lastUser)
  const lastUserIndex = messages.value.findIndex(message => message.id === lastUser.id)
  messages.value.splice(lastUserIndex)
  submitPrompt(retryText)
}

onMounted(async () => {
  let savedModel: ModelChoice | null = null
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || localStorage.getItem('qwen-local-chat-v1') || 'null') as {
      activeId?: string
      conversations?: Conversation[]
      selectedModel?: ModelChoice
    } | null
    if (isLocalModelChoice(saved?.selectedModel)) {
      savedModel = saved.selectedModel
      selectedModel.value = saved.selectedModel
    }
    if (saved?.conversations?.length) {
      conversations.value = saved.conversations
      activeId.value = saved.conversations.some(item => item.id === saved.activeId) ? saved.activeId! : saved.conversations[0]!.id
    } else {
      newConversation()
    }
  } catch {
    newConversation()
  }

  const detected = await detectHardware()
  await refreshCacheStates()
  if (!savedModel && detected) {
    const cachedRecommendation = cacheStates.value[detected.recommendedModel]
    const cachedFallback = LOCAL_MODEL_CHOICES.find(model => cacheStates.value[model])
    selectedModel.value = cachedRecommendation
      ? detected.recommendedModel
      : cachedFallback || detected.recommendedModel
    persist()
  }
  initialSelectionComplete.value = true
  if (detected?.webgpuAvailable !== false) await prepareModel(false)
})

onBeforeUnmount(() => {
  stopOperationTimer()
  void unloadQwenModel()
  void unloadLfmModel()
})
</script>

<template>
  <UDashboardGroup class="h-dvh">
    <UDashboardSidebar
      id="chat-sidebar"
      resizable
      collapsible
      :default-size="20"
      :min-size="16"
      :max-size="28"
      :collapsed-size="4"
    >
      <template #header="{ collapsed }">
        <div class="flex items-center gap-2">
          <UAvatar icon="i-lucide-bot" color="neutral" size="sm" />
          <span v-if="!collapsed" class="font-semibold">Local AI</span>
        </div>
      </template>

      <template #default="{ collapsed }">
        <ChatSidebar
          :conversations="conversationSummaries"
          :active-id="activeId"
          :collapsed="collapsed"
          @create="newConversation"
          @select="selectConversation"
          @remove="removeConversation"
        />
      </template>
    </UDashboardSidebar>

    <UDashboardPanel id="chat-panel">
      <template #header>
        <UDashboardNavbar :title="activeConversation?.title || '새 채팅'">
          <template #right>
            <select
              v-model="selectedModel"
              class="max-w-56 rounded-md border border-default bg-default px-2 py-1.5 text-sm text-highlighted"
              aria-label="로컬 모델 선택"
              :disabled="isBusy"
              @change="handleModelChange"
            >
              <option v-for="item in modelOptions" :key="item.value" :value="item.value">
                {{ item.label }}
              </option>
            </select>
            <UButton
              label="모델 관리"
              icon="i-lucide-boxes"
              color="neutral"
              variant="outline"
              @click="modelManagerOpen = true; refreshCacheStates()"
            />
            <UBadge :color="modelBadgeColor" variant="subtle">
              {{ modelReady ? `${modelLabel} 준비됨` : modelState === 'loading' ? `준비 ${modelPercent}%` : modelState === 'checking' ? '확인 중' : '점검 필요' }}
            </UBadge>
            <UButton
              :icon="colorMode.value === 'dark' ? 'i-lucide-sun' : 'i-lucide-moon'"
              color="neutral"
              variant="ghost"
              aria-label="색상 모드 전환"
              @click="colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'"
            />
          </template>
        </UDashboardNavbar>
      </template>

      <template #body>
        <div v-if="messages.length === 0" class="mx-auto flex min-h-full w-full max-w-2xl flex-col items-center justify-center gap-6 py-8 text-center">
          <UAvatar icon="i-lucide-bot" color="neutral" size="xl" />

          <div class="space-y-2">
            <h2 class="text-3xl font-semibold tracking-tight text-highlighted">
              무엇을 도와드릴까요?
            </h2>
            <p class="text-muted">
              {{ modelDescription }} 대화와 추론은 기기 안에서 처리됩니다.
            </p>
          </div>

          <UCard v-if="profile || profiling || profileError" class="w-full text-left">
            <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div class="min-w-0 space-y-1">
                <div class="flex flex-wrap items-center gap-2">
                  <p class="font-medium text-highlighted">기기 맞춤 모델</p>
                  <UBadge v-if="profile" color="primary" variant="subtle">
                    추천 {{ recommendedModelLabel }}
                  </UBadge>
                </div>
                <p class="text-sm text-muted">{{ hardwareSummary }}</p>
                <p v-if="profile" class="text-sm text-muted">{{ profile.recommendationReason }}</p>
                <p v-if="profile && profile.confidence === 'low'" class="text-xs text-warning">
                  브라우저가 실제 VRAM을 공개하지 않아 안전한 쪽으로 판정했습니다.
                </p>
              </div>
              <UButton
                label="다른 모델 보기"
                icon="i-lucide-sliders-horizontal"
                color="neutral"
                variant="outline"
                :disabled="!initialSelectionComplete"
                @click="modelManagerOpen = true; refreshCacheStates()"
              />
            </div>
          </UCard>

          <UCard v-if="!modelReady" class="w-full text-left">
            <div class="space-y-3">
              <div class="flex items-center justify-between gap-4">
                <div class="min-w-0">
                  <p class="font-medium text-highlighted">{{ modelState === 'loading' ? operationStatusText : modelStatusText }}</p>
                  <p class="text-sm text-muted">{{ modelStorageHint }}</p>
                </div>
                <UBadge :color="modelBadgeColor" variant="subtle">{{ modelPercent }}%</UBadge>
              </div>

              <UProgress v-if="modelState !== 'error'" :model-value="modelPercent" />

              <UAlert
                v-if="modelState === 'error'"
                title="모델을 준비하지 못했습니다"
                :description="modelError || 'WebGPU 상태를 확인해 주세요.'"
                icon="i-lucide-circle-alert"
                color="error"
                variant="subtle"
              >
                <template #actions>
                  <UButton
                    label="다시 시도"
                    icon="i-lucide-refresh-cw"
                    color="error"
                    variant="outline"
                    :disabled="webgpuAvailable === false"
                    @click="prepareModel()"
                  />
                </template>
              </UAlert>
            </div>
          </UCard>

          <div class="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
            <UButton
              v-for="item in starterPrompts"
              :key="item.label"
              :label="item.label"
              :icon="item.icon"
              color="neutral"
              variant="outline"
              size="lg"
              block
              class="justify-start"
              :disabled="!modelReady"
              @click="submitPrompt(item.label)"
            />
          </div>
        </div>

        <UChatMessages
          v-else
          :messages="messages"
          :status="chatStatus"
          should-auto-scroll
          :spacing-offset="24"
          :user="{ avatar: { icon: 'i-lucide-user-round' } }"
          :assistant="{ avatar: { icon: 'i-lucide-bot' } }"
          class="mx-auto w-full max-w-3xl"
        >
          <template #indicator>
            <UChatShimmer :text="operationStatusText || '답변을 정리하고 있습니다'" />
          </template>

          <template #content="{ message }">
            <div class="space-y-3">
              <UChatShimmer
                v-if="message.role === 'assistant' && chatStatus === 'streaming' && message.id === messages.at(-1)?.id && !textById(message.id) && !messageById(message.id)?.reasoning"
                :text="operationStatusText || '답변 생성 중'"
              />
              <UChatReasoning
                v-if="message.role === 'assistant' && messageById(message.id)?.reasoning"
                :text="messageById(message.id)?.reasoning"
                :streaming="chatStatus === 'streaming' && message.id === messages.at(-1)?.id && !textById(message.id)"
              />
              <p
                v-for="(part, index) in textPartsById(message.id)"
                :key="`${message.id}-${index}`"
                class="whitespace-pre-wrap"
              >
                {{ part.text }}
              </p>
              <div
                v-if="messageById(message.id)?.model || messageById(message.id)?.metrics"
                class="flex flex-wrap gap-2 text-xs text-muted"
              >
                <span v-if="modelLabelByMessageId(message.id)">{{ modelLabelByMessageId(message.id) }}</span>
                <span v-if="messageById(message.id)?.metrics?.decodeTokensPerSecond">{{ messageById(message.id)!.metrics!.decodeTokensPerSecond!.toFixed(1) }} tok/s</span>
                <span v-if="messageById(message.id)?.metrics?.timeToFirstVisibleTokenSeconds">연산 시작 {{ messageById(message.id)!.metrics!.timeToFirstTokenSeconds?.toFixed(1) }}초</span>
                <span v-if="messageById(message.id)?.metrics?.timeToFirstVisibleTokenSeconds">답변 시작 {{ messageById(message.id)!.metrics!.timeToFirstVisibleTokenSeconds!.toFixed(1) }}초</span>
                <span v-else-if="messageById(message.id)?.metrics?.timeToFirstTokenSeconds">첫 토큰 {{ messageById(message.id)!.metrics!.timeToFirstTokenSeconds!.toFixed(1) }}초</span>
              </div>
            </div>
          </template>

          <template #actions="{ message }">
            <UButton
              v-if="message.role === 'assistant'"
              :icon="copiedMessageId === message.id ? 'i-lucide-check' : 'i-lucide-copy'"
              :label="copiedMessageId === message.id ? '복사됨' : '복사'"
              color="neutral"
              variant="ghost"
              size="xs"
              @click="copyMessageById(message.id)"
            />
          </template>
        </UChatMessages>
      </template>

      <template #footer>
        <div class="mx-auto w-full max-w-3xl space-y-2">
          <UAlert
            v-if="lastError"
            color="error"
            variant="subtle"
            icon="i-lucide-circle-alert"
            title="응답을 만들지 못했습니다"
            :description="`${lastError} 모델 상태를 확인한 뒤 다시 보내 주세요.`"
            close
            @update:open="lastError = null"
          />

          <UChatPrompt
            v-model="prompt"
            :disabled="!modelReady"
            :error="chatStatus === 'error' ? new Error(lastError || '응답 생성 실패') : undefined"
            :placeholder="modelReady ? '메시지를 입력하세요' : modelState === 'error' ? 'WebGPU 상태를 확인해 주세요' : `모델 준비 중 · ${modelPercent}%`"
            color="neutral"
            variant="subtle"
            :rows="1"
            :maxrows="6"
            @submit="submitPrompt()"
          >
            <template #footer>
              <div class="flex min-w-0 items-center gap-2 text-xs text-muted">
                <UIcon :name="modelReady ? 'i-lucide-shield-check' : 'i-lucide-download'" />
                <span class="truncate">{{ modelReady ? runtimeLabel : modelStatusText }}</span>
              </div>
              <UChatPromptSubmit
                :status="chatStatus"
                :disabled="!prompt.trim() || !modelReady"
                color="neutral"
                @stop="stop"
                @reload="retryLast"
              />
            </template>
          </UChatPrompt>

          <p class="text-center text-xs text-muted">
            AI는 실수할 수 있습니다. 중요한 정보는 다시 확인하세요.
          </p>
        </div>
      </template>
    </UDashboardPanel>
  </UDashboardGroup>

  <div
    v-if="modelManagerOpen"
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="로컬 모델 관리"
    @click.self="modelManagerOpen = false"
  >
    <UCard class="max-h-[92vh] w-full max-w-4xl overflow-y-auto">
      <template #header>
        <div class="flex items-start justify-between gap-4">
          <div>
            <h2 class="text-xl font-semibold text-highlighted">이 PC의 로컬 모델</h2>
            <p class="mt-1 text-sm text-muted">추천 모델 하나만 먼저 받고, 나머지는 필요할 때 선택합니다.</p>
          </div>
          <UButton
            icon="i-lucide-x"
            color="neutral"
            variant="ghost"
            aria-label="모델 관리 닫기"
            @click="modelManagerOpen = false"
          />
        </div>
      </template>

      <div class="space-y-5">
        <div class="rounded-lg border border-default bg-elevated/40 p-4">
          <div class="flex flex-wrap items-center gap-2">
            <UBadge color="primary" variant="subtle">자동 추천 {{ recommendedModelLabel || '분석 중' }}</UBadge>
            <UBadge color="neutral" variant="subtle">저장된 모델 {{ cachedCount }}개</UBadge>
            <UBadge v-if="cacheChecking" color="warning" variant="subtle">캐시 확인 중</UBadge>
          </div>
          <p class="mt-2 text-sm text-muted">{{ hardwareSummary }}</p>
          <p v-if="profile" class="mt-1 text-sm text-muted">{{ profile.recommendationReason }}</p>
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          <article
            v-for="model in LOCAL_MODEL_CHOICES"
            :key="model"
            class="flex flex-col gap-3 rounded-lg border border-default p-4"
            :class="selectedModel === model ? 'ring-2 ring-primary/50' : ''"
          >
            <div class="flex items-start justify-between gap-3">
              <div>
                <h3 class="font-semibold text-highlighted">{{ LOCAL_MODELS[model].label }}</h3>
                <p class="text-xs text-muted">{{ LOCAL_MODELS[model].runtime }}</p>
              </div>
              <div class="flex flex-wrap justify-end gap-1">
                <UBadge v-if="profile?.recommendedModel === model" color="primary" variant="subtle">추천</UBadge>
                <UBadge v-if="selectedModel === model && modelState === 'loading'" color="warning" variant="subtle">준비 {{ modelPercent }}%</UBadge>
                <UBadge v-if="cacheStates[model]" color="success" variant="subtle">저장됨</UBadge>
                <UBadge v-if="!LOCAL_MODELS[model].autoEligible" color="neutral" variant="subtle">선택 다운로드</UBadge>
              </div>
            </div>

            <p class="text-sm text-muted">{{ LOCAL_MODELS[model].description }}</p>
            <p v-if="LOCAL_MODELS[model].caveat" class="text-xs text-warning">{{ LOCAL_MODELS[model].caveat }}</p>

            <dl class="grid grid-cols-2 gap-2 text-xs text-muted">
              <div>
                <dt>다운로드</dt>
                <dd class="font-medium text-highlighted">약 {{ LOCAL_MODELS[model].estimatedDownloadGb.toFixed(2) }}GB</dd>
              </div>
              <div>
                <dt>예상 VRAM</dt>
                <dd class="font-medium text-highlighted">약 {{ (LOCAL_MODELS[model].estimatedVramMb / 1024).toFixed(1) }}GB</dd>
              </div>
            </dl>

            <div class="mt-auto flex flex-wrap items-center gap-2">
              <UButton
                :label="selectedModel === model && modelState === 'loading' ? `준비 ${modelPercent}%` : modelReady && selectedModel === model ? '사용 중' : cacheStates[model] ? '캐시에서 실행' : '다운로드하고 실행'"
                :icon="cacheStates[model] ? 'i-lucide-play' : 'i-lucide-download'"
                :color="profile?.recommendedModel === model ? 'primary' : 'neutral'"
                :variant="profile?.recommendedModel === model ? 'solid' : 'outline'"
                :disabled="isBusy || (modelReady && selectedModel === model)"
                @click="activateManagedModel(model)"
              />
              <UButton
                v-if="cacheStates[model]"
                label="삭제"
                icon="i-lucide-trash-2"
                color="error"
                variant="ghost"
                :loading="cacheActionModel === model"
                :disabled="isBusy || cacheActionModel !== null"
                @click="removeCachedModel(model)"
              />
              <a
                :href="LOCAL_MODELS[model].source"
                target="_blank"
                rel="noreferrer"
                class="ml-auto text-xs text-muted underline underline-offset-2"
              >원본 정보</a>
            </div>
          </article>
        </div>

        <UAlert
          title="다운로드와 대화는 이 브라우저 안에서 처리됩니다"
          description="모델 파일은 브라우저 캐시에 저장되며 서버로 대화가 전송되지 않습니다. 브라우저 데이터 삭제 시 모델도 다시 받아야 합니다."
          icon="i-lucide-shield-check"
          color="neutral"
          variant="subtle"
        />
      </div>
    </UCard>
  </div>
</template>
