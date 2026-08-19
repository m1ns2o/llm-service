<script setup lang="ts">
type ChatStatus = 'ready' | 'submitted' | 'streaming' | 'error'

interface TextPart {
  type: 'text'
  text: string
}

interface LocalMessage {
  id: string
  role: 'user' | 'assistant'
  parts: TextPart[]
  reasoning?: string
  createdAt: string
  metrics?: {
    decodeTokensPerSecond?: number
    timeToFirstTokenSeconds?: number
  }
}

interface Conversation {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: LocalMessage[]
}

const STORAGE_KEY = 'qwen-local-chat-v1'
const SYSTEM_PROMPT = '당신은 학생, 교사, 일반 사용자를 돕는 친절하고 정확한 한국어 AI입니다. 결론부터 간결하게 답하고, 학습 질문에는 이해 수준에 맞는 예시와 확인 질문을 사용하세요. 모르는 내용은 추측하지 말고 불확실성을 분명히 밝히세요.'
const MAX_CONTEXT_MESSAGES = 6
const MAX_CONTEXT_CHARACTERS = 1200
const DEFAULT_OUTPUT_TOKENS = 1536
const OUTPUT_TOKENS_STORAGE_KEY = 'qwen-output-tokens-v1'
const OUTPUT_TOKEN_OPTIONS = [
  { label: '보통 · 1,024 토큰', value: 1024 },
  { label: '길게 · 1,536 토큰', value: 1536 },
  { label: '최대 · 2,048 토큰', value: 2048 }
]
const APPEARANCE_OPTIONS = [
  { label: '시스템 설정', value: 'system' },
  { label: '라이트', value: 'light' },
  { label: '다크', value: 'dark' }
]

const toast = useToast()
const colorMode = useColorMode()
const prompt = ref('')
const conversations = ref<Conversation[]>([])
const activeId = ref('')
const chatStatus = ref<ChatStatus>('ready')
const lastError = ref<string | null>(null)
const copiedMessageId = ref<string | null>(null)
const settingsOpen = ref(false)
const maxOutputTokens = ref(DEFAULT_OUTPUT_TOKENS)

const {
  modelOptions,
  selectedModelId,
  selectedModelLabel,
  modelState,
  modelProgress,
  modelStatusText,
  modelError,
  webgpuAvailable,
  nativeRuntime,
  setNativeColorScheme,
  restoreModelSelection,
  selectModel,
  loadModel,
  beginGeneration,
  wasInterrupted,
  stopGeneration
} = useLocalQwen()

const activeConversation = computed(() => conversations.value.find(item => item.id === activeId.value))
const messages = computed(() => activeConversation.value?.messages || [])
const conversationSummaries = computed(() => [...conversations.value]
  .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  .map(({ id, title, updatedAt }) => ({ id, title, updatedAt })))
const isBusy = computed(() => chatStatus.value === 'submitted' || chatStatus.value === 'streaming' || modelState.value === 'loading')
const modelReady = computed(() => modelState.value === 'ready')
const modelPercent = computed(() => Math.round(modelProgress.value * 100))
const modelBadgeColor = computed(() => {
  if (modelState.value === 'ready') return 'success'
  if (modelState.value === 'error') return 'error'
  if (modelState.value === 'loading' || modelState.value === 'checking') return 'warning'
  return 'neutral'
})

function changeAppearance(value: unknown) {
  if (typeof value === 'string' && APPEARANCE_OPTIONS.some(option => option.value === value)) {
    colorMode.preference = value
  }
}

watch(() => colorMode.value, value => setNativeColorScheme(value === 'dark' ? 'dark' : 'light'), { immediate: true })

function changeOutputLength(value: unknown) {
  const parsed = Number(value)
  if (!OUTPUT_TOKEN_OPTIONS.some(option => option.value === parsed)) return
  maxOutputTokens.value = parsed
  if (import.meta.client) localStorage.setItem(OUTPUT_TOKENS_STORAGE_KEY, String(parsed))
}
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
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ activeId: activeId.value, conversations: conversations.value }))
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

async function allowBrowserPaint() {
  await nextTick()
  await new Promise<void>((resolve) => {
    const timeout = window.setTimeout(resolve, 50)
    window.requestAnimationFrame(() => {
      window.clearTimeout(timeout)
      resolve()
    })
  })
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
  try {
    await loadModel()
  } catch {
    if (notify) {
      toast.add({ title: '모델 준비 실패', description: modelError.value || '모델을 불러오지 못했습니다.', color: 'error' })
    }
  }
}

async function changeModel(value: unknown) {
  if (typeof value !== 'string' || isBusy.value || value === selectedModelId.value) return
  lastError.value = null
  try {
    await selectModel(value)
    toast.add({ title: `${selectedModelLabel.value}로 변경했습니다` })
  } catch {
    toast.add({ title: '모델 변경 실패', description: modelError.value || '모델을 불러오지 못했습니다.', color: 'error' })
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
  persist()

  let assistant: LocalMessage = {
    id: makeId(),
    role: 'assistant',
    parts: [{ type: 'text', text: '' }],
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
    const engine = await loadModel()
    chatStatus.value = 'streaming'
    beginGeneration()

    const requestMessages = [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      ...recentRequestMessages(conversation, assistant.id)
    ]

    const chunks = await engine.chat.completions.create({
      messages: requestMessages,
      temperature: 1,
      top_p: 1,
      presence_penalty: 2,
      max_tokens: maxOutputTokens.value,
      extra_body: { enable_thinking: false },
      stream: true,
      stream_options: { include_usage: true }
    })

    let raw = ''
    let usageExtra: { decode_tokens_per_s?: number, time_to_first_token_s?: number } | undefined
    for await (const chunk of chunks) {
      if (wasInterrupted()) break
      const delta = chunk.choices[0]?.delta.content || ''
      raw += delta
      const parsed = parseOutput(raw)
      replaceAssistant({
        reasoning: parsed.reasoning,
        parts: [{ type: 'text', text: parsed.answer }]
      })
      if (chunk.usage?.extra) usageExtra = chunk.usage.extra
      if (delta) await allowBrowserPaint()
    }

    if (!textOf(assistant).trim() && !wasInterrupted()) {
      replaceAssistant({ parts: [{ type: 'text', text: '답변을 완성하기 전에 출력 한도에 도달했습니다. 질문을 더 짧게 나누어 다시 시도해 주세요.' }] })
    }
    if (usageExtra) {
      replaceAssistant({
        metrics: {
          decodeTokensPerSecond: usageExtra.decode_tokens_per_s,
          timeToFirstTokenSeconds: usageExtra.time_to_first_token_s
        }
      })
    }
    chatStatus.value = 'ready'
  } catch (error: unknown) {
    lastError.value = error instanceof Error ? error.message : String(error)
    replaceAssistant({ parts: [{ type: 'text', text: '응답을 생성하지 못했습니다. 모델 상태를 확인한 뒤 다시 시도해 주세요.' }] })
    chatStatus.value = 'error'
  } finally {
    conversation.updatedAt = new Date().toISOString()
    persist()
  }
}

async function stop() {
  await stopGeneration()
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

function textById(id: string) {
  const message = messageById(id)
  return message ? textOf(message) : ''
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

onMounted(() => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') as { activeId?: string, conversations?: Conversation[] } | null
    if (saved?.conversations?.length) {
      conversations.value = saved.conversations
      activeId.value = saved.conversations.some(item => item.id === saved.activeId) ? saved.activeId! : saved.conversations[0]!.id
    } else {
      newConversation()
    }
  } catch {
    newConversation()
  }

  const savedOutputTokens = Number(localStorage.getItem(OUTPUT_TOKENS_STORAGE_KEY))
  if (OUTPUT_TOKEN_OPTIONS.some(option => option.value === savedOutputTokens)) {
    maxOutputTokens.value = savedOutputTokens
  }

  restoreModelSelection()
  void prepareModel(false)
})
</script>

<template>
  <UDashboardGroup class="app-shell">
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
        <div
          class="flex w-full min-w-0 items-center"
          :class="collapsed ? 'justify-center' : 'gap-2'"
        >
          <UAvatar v-if="!collapsed" icon="i-lucide-bot" color="neutral" size="sm" />
          <span v-if="!collapsed" class="font-semibold">Qwen Local</span>
          <UDashboardSidebarCollapse :class="collapsed ? undefined : 'ms-auto'" />
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
            <UBadge :color="modelBadgeColor" variant="subtle" class="max-w-32 truncate sm:max-w-none">
              {{ modelReady ? '모델 준비됨' : modelState === 'loading' ? `다운로드 ${modelPercent}%` : modelState === 'checking' ? '확인 중' : '점검 필요' }}
            </UBadge>
            <UModal v-model:open="settingsOpen" title="설정" description="화면과 답변 생성 방식을 조정합니다.">
              <UButton
                icon="i-lucide-settings-2"
                color="neutral"
                variant="ghost"
                class="mobile-touch-target"
                aria-label="설정 열기"
              />

              <template #body>
                <div class="space-y-5">
                  <UFormField label="화면 모드" description="기본값은 기기의 시스템 설정을 따릅니다.">
                    <USelect
                      :model-value="colorMode.preference"
                      :items="APPEARANCE_OPTIONS"
                      class="mt-2 w-full"
                      aria-label="화면 모드"
                      @update:model-value="changeAppearance"
                    />
                  </UFormField>

                  <UFormField label="답변 길이" description="긴 답변일수록 생성 시간과 배터리 사용량이 늘어납니다.">
                    <USelect
                      :model-value="maxOutputTokens"
                      :items="OUTPUT_TOKEN_OPTIONS"
                      class="mt-2 w-full"
                      aria-label="최대 출력 토큰"
                      @update:model-value="changeOutputLength"
                    />
                  </UFormField>

                  <div class="rounded-lg bg-elevated/60 p-3 text-sm text-muted">
                    현재 {{ selectedModelLabel }} · 최대 {{ maxOutputTokens.toLocaleString() }}토큰
                  </div>
                </div>
              </template>
            </UModal>
          </template>
        </UDashboardNavbar>
      </template>

      <template #body>
        <div v-if="messages.length === 0" class="chat-empty-state mx-auto flex min-h-full w-full max-w-3xl flex-col justify-center py-6 sm:py-12">
          <div class="mb-8 space-y-2 text-center sm:mb-10">
            <h1 class="text-3xl font-semibold tracking-tight text-highlighted sm:text-4xl">
              무엇을 도와드릴까요?
            </h1>
            <p class="text-pretty text-muted">
              {{ selectedModelLabel }}가 {{ nativeRuntime ? '기기의 네이티브 GPU' : '이 브라우저' }}에서 직접 답합니다.
            </p>
          </div>

          <UCard v-if="!modelReady" class="mb-4 w-full">
            <div class="space-y-3">
              <div class="flex items-center justify-between gap-4">
                <div class="min-w-0">
                  <p class="font-medium text-highlighted">초기 설정 진행 중</p>
                  <p class="text-sm text-muted">모델을 이 기기에 준비하고 있습니다. 다음 실행부터는 저장된 파일을 사용합니다.</p>
                </div>
                <UBadge :color="modelBadgeColor" variant="subtle">{{ modelPercent }}%</UBadge>
              </div>

              <UProgress v-if="modelState !== 'error'" :model-value="modelPercent" />

              <UAlert
                v-if="modelState === 'error'"
                title="모델을 준비하지 못했습니다"
                :description="modelError || '기기 가속 상태를 확인해 주세요.'"
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

          <UChatPrompt
            v-model="prompt"
            :disabled="!modelReady"
            :error="chatStatus === 'error' ? new Error(lastError || '응답 생성 실패') : undefined"
            :placeholder="modelReady ? '무엇이든 물어보세요' : modelState === 'error' ? '기기 가속 상태를 확인해 주세요' : `모델 준비 중 · ${modelPercent}%`"
            :rows="2"
            :maxrows="6"
            @submit="submitPrompt()"
          >
            <template #footer>
              <div class="flex min-w-0 items-center gap-1">
                <USelect
                  :model-value="selectedModelId"
                  :items="modelOptions"
                  variant="ghost"
                  size="sm"
                  class="w-28 shrink-0"
                  aria-label="모델 선택"
                  :disabled="isBusy"
                  @update:model-value="changeModel"
                />
                <div class="hidden min-w-0 items-center gap-1.5 text-xs text-muted sm:flex">
                  <UIcon :name="modelReady ? 'i-lucide-shield-check' : 'i-lucide-download'" />
                  <span class="truncate">{{ modelStatusText }}</span>
                </div>
              </div>
              <UChatPromptSubmit
                :status="chatStatus"
                :disabled="!prompt.trim() || !modelReady"
                class="chat-submit-button justify-center"
                @stop="stop"
                @reload="retryLast"
              />
            </template>
          </UChatPrompt>
        </div>

        <UChatMessages
          v-else
          :messages="messages"
          :status="chatStatus"
          should-auto-scroll
          :spacing-offset="24"
          :user="{ avatar: { icon: 'i-lucide-user-round' } }"
          :assistant="{ avatar: { icon: 'i-lucide-bot' } }"
          class="chat-message-list mx-auto w-full max-w-3xl"
        >
          <template #indicator>
            <UChatShimmer :text="modelState === 'loading' ? '모델을 준비하고 있습니다' : '답변을 정리하고 있습니다'" />
          </template>

          <template #content="{ message }">
            <div class="space-y-3">
              <UChatShimmer
                v-if="message.role === 'assistant' && chatStatus === 'streaming' && message.id === messages.at(-1)?.id && !textById(message.id) && !messageById(message.id)?.reasoning"
                text="답변 생성 중"
              />
              <UChatReasoning
                v-if="message.role === 'assistant' && messageById(message.id)?.reasoning"
                :text="messageById(message.id)?.reasoning"
                :streaming="chatStatus === 'streaming' && message.id === messages.at(-1)?.id && !textById(message.id)"
              />
              <MDC
                v-if="textById(message.id)"
                :value="textById(message.id)"
                tag="div"
                class="max-w-none"
              />
              <div v-if="messageById(message.id)?.metrics" class="flex flex-wrap gap-2 text-xs text-muted">
                <span v-if="messageById(message.id)?.metrics?.decodeTokensPerSecond">{{ messageById(message.id)!.metrics!.decodeTokensPerSecond!.toFixed(1) }} tok/s</span>
                <span v-if="messageById(message.id)?.metrics?.timeToFirstTokenSeconds">첫 토큰 {{ messageById(message.id)!.metrics!.timeToFirstTokenSeconds!.toFixed(1) }}초</span>
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

      <template v-if="messages.length > 0" #footer>
        <div class="chat-composer mx-auto w-full max-w-3xl space-y-2">
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
            :placeholder="modelReady ? '메시지를 입력하세요' : modelState === 'error' ? '기기 가속 상태를 확인해 주세요' : `모델 준비 중 · ${modelPercent}%`"
            :rows="1"
            :maxrows="6"
            @submit="submitPrompt()"
          >
            <template #footer>
              <div class="flex min-w-0 items-center gap-1">
                <USelect
                  :model-value="selectedModelId"
                  :items="modelOptions"
                  variant="ghost"
                  size="sm"
                  class="w-28 shrink-0"
                  aria-label="모델 선택"
                  :disabled="isBusy"
                  @update:model-value="changeModel"
                />
                <div class="hidden min-w-0 items-center gap-1.5 text-xs text-muted sm:flex">
                  <UIcon :name="modelReady ? 'i-lucide-shield-check' : 'i-lucide-download'" />
                  <span class="truncate">{{ modelStatusText }}</span>
                </div>
              </div>
              <UChatPromptSubmit
                :status="chatStatus"
                :disabled="!prompt.trim() || !modelReady"
                class="chat-submit-button justify-center"
                @stop="stop"
                @reload="retryLast"
              />
            </template>
          </UChatPrompt>

        </div>
      </template>
    </UDashboardPanel>
  </UDashboardGroup>
</template>
