<script setup lang="ts">
interface ConversationSummary {
  id: string
  title: string
  updatedAt: string
}

defineProps<{
  conversations: ConversationSummary[]
  activeId: string
  collapsed?: boolean
}>()

defineEmits<{
  select: [id: string]
  create: []
  remove: [id: string]
}>()
</script>

<template>
  <div class="flex h-full min-h-0 flex-col gap-3">
    <UButton
      :label="collapsed ? undefined : '새 채팅'"
      icon="i-lucide-square-pen"
      color="neutral"
      variant="ghost"
      block
      :square="collapsed"
      class="justify-start"
      @click="$emit('create')"
    />

    <p v-if="!collapsed" class="px-2 pt-2 text-xs font-medium text-muted">
      최근 대화
    </p>

    <div class="min-h-0 flex-1 space-y-1 overflow-y-auto">
      <div
        v-for="conversation in conversations"
        :key="conversation.id"
        class="flex items-center gap-1"
      >
        <UTooltip :text="conversation.title" :disabled="!collapsed">
          <UButton
            :label="collapsed ? undefined : conversation.title"
            icon="i-lucide-message-square"
            color="neutral"
            :variant="conversation.id === activeId ? 'soft' : 'ghost'"
            :square="collapsed"
            class="min-w-0 flex-1 justify-start"
            :ui="{ label: 'truncate' }"
            @click="$emit('select', conversation.id)"
          />
        </UTooltip>

        <UButton
          v-if="!collapsed"
          icon="i-lucide-trash-2"
          color="neutral"
          variant="ghost"
          size="sm"
          aria-label="대화 삭제"
          @click="$emit('remove', conversation.id)"
        />
      </div>

      <div v-if="conversations.length === 0 && !collapsed" class="py-8 text-center text-sm text-muted">
        저장된 대화가 없습니다.
      </div>
    </div>

    <div v-if="!collapsed" class="flex items-center gap-2 px-2 py-1 text-xs text-muted">
      <UIcon name="i-lucide-shield-check" class="size-4 shrink-0" />
      <span>기기 안에서 안전하게 실행</span>
    </div>
  </div>
</template>
