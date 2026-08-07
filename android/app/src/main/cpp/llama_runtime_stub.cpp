#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_statusNative(JNIEnv* env, jclass) {
    return env->NewStringUTF("blocked: llama.cpp source is not linked; use scripts/build_llama_runtime.sh");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_generateNative(JNIEnv* env, jclass, jstring) {
    return env->NewStringUTF("실행 차단: 고정된 llama.cpp Android JNI 런타임을 먼저 빌드하세요.");
}
