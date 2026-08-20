package com.example.llmbench;

final class NativeRuntime {
    interface TokenCallback { void onToken(String token); }
    private static boolean loaded;
    static void load() { try { System.loadLibrary("llama_runtime"); loaded = true; } catch (UnsatisfiedLinkError ignored) { loaded = false; } }
    static String status() { return loaded ? statusNative() : "blocked: native library unavailable"; }
    static String availableBackends() { return loaded ? availableBackendsNative() : "cpu-arm64"; }
    static String loadModel(String path, String backend, int threads, int contextSize) {
        return loaded ? loadModelNative(path, backend, threads, contextSize) : "blocked: native llama.cpp library unavailable";
    }
    static String generate(String[] roles, String[] contents, int maxTokens, float temperature, TokenCallback callback) {
        return loaded ? generateNative(roles, contents, maxTokens, temperature, callback) : "실행 차단: native llama.cpp library unavailable";
    }
    static void requestInterrupt() { if (loaded) requestInterruptNative(); }
    static void unload() { if (loaded) unloadNative(); }
    private static native String statusNative();
    private static native String availableBackendsNative();
    private static native String loadModelNative(String path, String backend, int threads, int contextSize);
    private static native String generateNative(String[] roles, String[] contents, int maxTokens, float temperature, TokenCallback callback);
    private static native void requestInterruptNative();
    private static native void unloadNative();
}
