package com.example.llmbench;

final class NativeRuntime {
    private static boolean loaded;
    static void load() { try { System.loadLibrary("llama_runtime"); loaded = true; } catch (UnsatisfiedLinkError ignored) { loaded = false; } }
    static String status() { return loaded ? statusNative() : "blocked: native library unavailable"; }
    static String generate(String prompt) { return loaded ? generateNative(prompt) : "실행 차단: native llama.cpp library unavailable"; }
    private static native String statusNative();
    private static native String generateNative(String prompt);
}
