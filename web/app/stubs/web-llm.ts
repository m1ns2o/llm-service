// Android always uses the JNI bridge. Keeping WebLLM out of this bundle avoids
// shipping and compiling a multi-megabyte browser-only worker in the APK.
export class WebWorkerMLCEngineHandler {
  onmessage() {
    throw new Error('WebLLM is unavailable in the Android native bundle')
  }
}

export async function CreateWebWorkerMLCEngine(): Promise<never> {
  throw new Error('WebLLM is unavailable in the Android native bundle')
}
