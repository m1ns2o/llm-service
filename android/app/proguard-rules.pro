# llama.cpp symbols are loaded through the small JNI boundary in debug builds.
-keepclassmembers class com.example.llmbench.MainActivity$AndroidLlmBridge {
    @android.webkit.JavascriptInterface <methods>;
}
-keep class com.example.llmbench.NativeRuntime { *; }
-keep interface com.example.llmbench.NativeRuntime$TokenCallback { *; }
