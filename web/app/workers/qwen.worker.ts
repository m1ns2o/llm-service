import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()

self.onmessage = handler.onmessage.bind(handler)
