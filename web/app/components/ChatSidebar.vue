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
  <div class="flex h-full min-h-0 flex-col gap-4">
    <UButton
      :label="collapsed ? undefined : '새 채팅'"
      icon="i-lucide-square-pen"
      color="neutral"
      variant="outline"
      block
      :square="collapsed"
      @click="$emit('create')"
    />

    <USeparator v-if="!collapsed" label="최근 대화" />

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

    <UAlert
      v-if="!collapsed"
      title="기기 안에서 실행"
      description="대화가 서버로 전송되지 않습니다."
      icon="i-lucide-shield-check"
      color="neutral"
      variant="subtle"
    />
  </div>
</template>
