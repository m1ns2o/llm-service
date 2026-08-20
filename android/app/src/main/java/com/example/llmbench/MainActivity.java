package com.example.llmbench;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.pm.ApplicationInfo;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.StatFs;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.webkit.WebViewAssetLoader;
import androidx.webkit.WebViewClientCompat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final String APP_ORIGIN = "https://appassets.androidplatform.net";
    private final ExecutorService nativeExecutor = Executors.newSingleThreadExecutor();
    private final Map<String, ModelSpec> models = new HashMap<>();
    private WebView webView;
    private String activeModelPath;
    private String activeBackend = "unloaded";
    private List<String> activeBackendOrder = new ArrayList<>();
    private int activeThreads = 1;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        NativeRuntime.load();
        loadModels();
        buildWebUi();
    }

    private void buildWebUi() {
        WebViewAssetLoader assetLoader = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();

        webView = new WebView(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(false);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        webView.setBackgroundColor(0xFFFFFFFF);
        WebView.setWebContentsDebuggingEnabled(
                (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0);
        webView.addJavascriptInterface(new AndroidLlmBridge(), "AndroidLLM");
        webView.setWebViewClient(new WebViewClientCompat() {
            @Override public WebResourceResponse shouldInterceptRequest(@NonNull WebView view, @NonNull WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }

            @Override @SuppressWarnings("deprecation")
            public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                return assetLoader.shouldInterceptRequest(Uri.parse(url));
            }

            @Override public boolean shouldOverrideUrlLoading(@NonNull WebView view, @NonNull WebResourceRequest request) {
                Uri uri = request.getUrl();
                return !"appassets.androidplatform.net".equals(uri.getHost());
            }
        });
        setContentView(webView);
        loadIndexAtRouteRoot();
    }

    private void loadIndexAtRouteRoot() {
        try (InputStream input = getAssets().open("web/index.html")) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int count;
            while ((count = input.read(chunk)) >= 0) if (count > 0) buffer.write(chunk, 0, count);
            // Loading index.html as a URL exposes `/index.html` to Vue Router
            // and incorrectly selects Nuxt's 404 route. A directory base URL
            // keeps the secure appassets origin while presenting `/` to the
            // app configured with baseURL=/assets/web/.
            webView.loadDataWithBaseURL(
                    APP_ORIGIN + "/assets/web/",
                    buffer.toString(StandardCharsets.UTF_8.name()),
                    "text/html", "UTF-8", null);
        } catch (Exception error) {
            throw new IllegalStateException("Android web UI load failed", error);
        }
    }

    private void loadModels() {
        try (InputStream input = getAssets().open("models.json")) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int count;
            while ((count = input.read(chunk)) >= 0) if (count > 0) buffer.write(chunk, 0, count);
            JSONArray entries = new JSONObject(buffer.toString(StandardCharsets.UTF_8.name())).getJSONArray("models");
            for (int i = 0; i < entries.length(); i++) {
                JSONObject item = entries.getJSONObject(i);
                if (!item.optBoolean("android_native", false)) continue;
                JSONObject artifact = item.getJSONArray("artifacts").getJSONObject(0);
                ModelSpec spec = new ModelSpec(
                        item.getString("id"), item.getString("name"), artifact.getString("filename"),
                        artifact.getString("url"), artifact.getString("sha256"), artifact.getLong("size_bytes"),
                        artifact.getString("quantization"), item.optDouble("minimum_ram_gb", 0));
                models.put(spec.id, spec);
            }
        } catch (Exception error) {
            throw new IllegalStateException("Android model manifest load failed", error);
        }
    }

    private File modelDirectory() {
        File external = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        return external != null ? external : new File(getFilesDir(), "models");
    }

    private boolean hasVulkan() {
        return Build.VERSION.SDK_INT >= 24
                && getPackageManager().hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL);
    }

    private boolean isQualcommDevice() {
        String hardware = Build.HARDWARE == null ? "" : Build.HARDWARE.toLowerCase(Locale.US);
        String board = Build.BOARD == null ? "" : Build.BOARD.toLowerCase(Locale.US);
        String socManufacturer = Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER != null
                ? Build.SOC_MANUFACTURER.toLowerCase(Locale.US) : "";
        String socModel = Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL != null
                ? Build.SOC_MODEL.toLowerCase(Locale.US) : "";
        return hardware.contains("qcom") || board.contains("qcom") || board.contains("sun")
                || socManufacturer.contains("qualcomm") || socModel.contains("snapdragon")
                || socModel.startsWith("sm");
    }

    private List<String> preferredBackendOrder() {
        List<String> result = new ArrayList<>();
        if (isQualcommDevice()) {
            // Adreno's optimized OpenCL kernels are fastest on verified Snapdragon devices.
            result.add("opencl");
            if (hasVulkan()) result.add("vulkan");
        } else {
            // Mali, Xclipse and PowerVR use portable Vulkan first. The OpenCL
            // probe is retained for devices exposing a compatible generic backend.
            if (hasVulkan()) result.add("vulkan");
            result.add("opencl");
        }
        result.add("cpu-arm64");
        return result;
    }

    private String backendPolicy(List<String> order) {
        return String.join(",", order);
    }

    private List<String> remainingBackends(List<String> order, String usedBackend) {
        int current = order.indexOf(usedBackend);
        if (current < 0 || current + 1 >= order.size()) return new ArrayList<>();
        return new ArrayList<>(order.subList(current + 1, order.size()));
    }

    private String actualBackend(String status, String fallback) {
        int marker = status.indexOf("backend=");
        if (marker < 0) return fallback;
        int start = marker + "backend=".length();
        int end = status.indexOf(" ·", start);
        return (end < 0 ? status.substring(start) : status.substring(start, end)).trim();
    }

    private ModelSpec resolveModel(String webModelId) {
        if (webModelId.contains("0.8B")) return models.get("qwen35-0.8b-android");
        if (webModelId.contains("4B")) return models.get("qwen35-4b-android");
        return models.get("qwen35-2b-android");
    }

    private void dispatch(JSONObject event) {
        final String encoded = JSONObject.quote(event.toString());
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.__nativeLlmDispatch && window.__nativeLlmDispatch(" + encoded + ")", null));
    }

    private void dispatch(String type, String requestId, JSONObject data) {
        try {
            JSONObject event = new JSONObject();
            event.put("type", type);
            event.put("requestId", requestId);
            event.put("data", data == null ? new JSONObject() : data);
            dispatch(event);
        } catch (Exception ignored) {}
    }

    private void dispatchError(String requestId, Throwable error) {
        try {
            dispatch("error", requestId, new JSONObject().put("message",
                    error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage()));
        } catch (Exception ignored) {}
    }

    final class AndroidLlmBridge {
        @JavascriptInterface public void setColorScheme(String scheme) {
            runOnUiThread(() -> {
                boolean dark = "dark".equals(scheme);
                int surface = Color.parseColor(dark ? "#111827" : "#FFFFFF");
                webView.setBackgroundColor(surface);
                getWindow().setStatusBarColor(surface);
                getWindow().setNavigationBarColor(surface);
                int visibility = getWindow().getDecorView().getSystemUiVisibility();
                if (dark) {
                    visibility &= ~View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
                    visibility &= ~View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
                } else {
                    visibility |= View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
                    visibility |= View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
                }
                getWindow().getDecorView().setSystemUiVisibility(visibility);
            });
        }

        @JavascriptInterface public String getCapabilities() {
            try {
                ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
                ((ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(memory);
                JSONObject result = new JSONObject();
                result.put("native", true);
                result.put("runtime", "llama.cpp JNI");
                List<String> backendOrder = preferredBackendOrder();
                result.put("backend", backendOrder.get(0));
                result.put("backendOrder", new JSONArray(backendOrder));
                JSONArray nativeBackends = new JSONArray();
                for (String backend : NativeRuntime.availableBackends().split(",")) {
                    if (!backend.isEmpty()) nativeBackends.put(backend);
                }
                result.put("nativeBackends", nativeBackends);
                result.put("ramMb", memory.totalMem / (1024 * 1024));
                result.put("device", Build.MANUFACTURER + " " + Build.MODEL);
                result.put("hardware", Build.HARDWARE == null ? "" : Build.HARDWARE);
                result.put("board", Build.BOARD == null ? "" : Build.BOARD);
                if (Build.VERSION.SDK_INT >= 31) {
                    result.put("socManufacturer", Build.SOC_MANUFACTURER == null ? "" : Build.SOC_MANUFACTURER);
                    result.put("socModel", Build.SOC_MODEL == null ? "" : Build.SOC_MODEL);
                }
                result.put("vulkan", hasVulkan());
                JSONArray available = new JSONArray();
                for (ModelSpec spec : models.values()) available.put(spec.id);
                result.put("models", available);
                return result.toString();
            } catch (Exception error) {
                return "{\"native\":false}";
            }
        }

        @JavascriptInterface public void prepareModel(String requestId, String webModelId) {
            nativeExecutor.execute(() -> {
                try {
                    ModelSpec spec = resolveModel(webModelId);
                    if (spec == null) throw new IllegalArgumentException("지원하지 않는 모델입니다.");
                    if (!Preflight.canRun(MainActivity.this, spec, modelDirectory())) {
                        throw new IllegalStateException("기기 RAM 또는 저장공간이 부족합니다.");
                    }
                    File file = ModelDownloader.download(spec, modelDirectory(), fraction -> {
                        try {
                            dispatch("model-progress", requestId, new JSONObject()
                                    .put("progress", fraction).put("label", spec.name));
                        } catch (Exception ignored) {}
                    });
                    List<String> backendOrder = preferredBackendOrder();
                    int cores = Runtime.getRuntime().availableProcessors();
                    int threads = Math.max(1, Math.min(8, cores - 2));
                    String status = NativeRuntime.loadModel(
                            file.getAbsolutePath(), backendPolicy(backendOrder), threads, 4096);
                    if (!status.startsWith("ready:")) throw new IllegalStateException(status);
                    String loadedBackend = actualBackend(status, backendOrder.get(0));
                    activeModelPath = file.getAbsolutePath();
                    activeBackendOrder = backendOrder;
                    activeBackend = loadedBackend;
                    activeThreads = threads;
                    dispatch("model-ready", requestId, new JSONObject()
                            .put("status", status).put("backend", loadedBackend)
                            .put("attemptedBackends", new JSONArray(backendOrder)).put("cached", file.exists()));
                } catch (Throwable error) {
                    dispatchError(requestId, error);
                }
            });
        }

        @JavascriptInterface public void generate(String requestId, String requestJson) {
            nativeExecutor.execute(() -> {
                try {
                    JSONObject request = new JSONObject(requestJson);
                    JSONArray messages = request.getJSONArray("messages");
                    List<String> roles = new ArrayList<>();
                    List<String> contents = new ArrayList<>();
                    for (int i = 0; i < messages.length(); i++) {
                        JSONObject message = messages.getJSONObject(i);
                        roles.add(message.getString("role"));
                        contents.add(message.getString("content"));
                    }
                    int maxTokens = Math.min(2048, request.optInt("maxTokens", 1536));
                    float temperature = (float) request.optDouble("temperature", 0.7);
                    long started = System.nanoTime();
                    final int[] tokenPieces = {0};
                    String result;
                    while (true) {
                        result = NativeRuntime.generate(
                                roles.toArray(new String[0]), contents.toArray(new String[0]),
                                maxTokens, temperature, token -> {
                                    tokenPieces[0]++;
                                    try { dispatch("token", requestId, new JSONObject().put("text", token)); }
                                    catch (Exception ignored) {}
                                });
                        if (tokenPieces[0] > 0 || !result.startsWith("실행 실패")) break;
                        List<String> remaining = remainingBackends(activeBackendOrder, activeBackend);
                        if (activeModelPath == null || remaining.isEmpty()) break;
                        String status = NativeRuntime.loadModel(
                                activeModelPath, backendPolicy(remaining), activeThreads, 4096);
                        if (!status.startsWith("ready:")) break;
                        String fallbackBackend = actualBackend(status, remaining.get(0));
                        if (fallbackBackend.equals(activeBackend)) break;
                        activeBackendOrder = remaining;
                        activeBackend = fallbackBackend;
                    }
                    long elapsedMs = (System.nanoTime() - started) / 1_000_000;
                    if (tokenPieces[0] == 0 && result.startsWith("실행")) {
                        throw new IllegalStateException(result);
                    }
                    dispatch("done", requestId, new JSONObject()
                            .put("elapsedMs", elapsedMs).put("tokenPieces", tokenPieces[0])
                            .put("backend", activeBackend));
                } catch (Throwable error) {
                    dispatchError(requestId, error);
                }
            });
        }

        @JavascriptInterface public void interrupt() {
            NativeRuntime.requestInterrupt();
        }
    }

    @Override protected void onDestroy() {
        NativeRuntime.requestInterrupt();
        nativeExecutor.shutdownNow();
        NativeRuntime.unload();
        if (webView != null) webView.destroy();
        super.onDestroy();
    }

    static final class ModelSpec {
        final String id, name, filename, url, sha256, quantization;
        final long sizeBytes;
        final double minRamGb;
        ModelSpec(String id, String name, String filename, String url, String sha256,
                  long sizeBytes, String quantization, double minRamGb) {
            this.id = id; this.name = name; this.filename = filename; this.url = url;
            this.sha256 = sha256; this.sizeBytes = sizeBytes; this.quantization = quantization;
            this.minRamGb = minRamGb;
        }
    }

    interface Progress { void update(double fraction); }

    static final class Preflight {
        static boolean canRun(Context context, ModelSpec model, File directory) {
            ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
            ((ActivityManager) context.getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(memory);
            if (!directory.exists() && !directory.mkdirs()) return false;
            StatFs stats = new StatFs(directory.getPath());
            long requiredRam = (long) (model.minRamGb * 1_000_000_000L);
            return memory.totalMem >= requiredRam
                    && stats.getAvailableBytes() >= model.sizeBytes + model.sizeBytes / 10;
        }
    }

    static final class ModelDownloader {
        static File download(ModelSpec model, File directory, Progress progress) throws Exception {
            if (!directory.exists() && !directory.mkdirs()) throw new IllegalStateException("모델 폴더 생성 실패");
            File complete = new File(directory, model.filename);
            File partial = new File(directory, model.filename + ".partial");
            if (complete.exists() && complete.length() == model.sizeBytes) {
                progress.update(1);
                return complete;
            }
            long offset = partial.exists() ? partial.length() : 0;
            HttpURLConnection connection = (HttpURLConnection) new URL(model.url).openConnection();
            connection.setInstanceFollowRedirects(true);
            connection.setConnectTimeout(30_000);
            connection.setReadTimeout(60_000);
            if (offset > 0) connection.setRequestProperty("Range", "bytes=" + offset + "-");
            int code = connection.getResponseCode();
            if (code != 200 && code != 206) throw new IllegalStateException("모델 다운로드 HTTP " + code);
            boolean append = offset > 0 && code == 206;
            if (!append) { offset = 0; if (partial.exists() && !partial.delete()) throw new IllegalStateException("부분 파일 초기화 실패"); }
            try (InputStream input = new BufferedInputStream(connection.getInputStream());
                 OutputStream output = new FileOutputStream(partial, append)) {
                byte[] buffer = new byte[1024 * 1024];
                long received = offset;
                int count;
                while ((count = input.read(buffer)) >= 0) {
                    if (count == 0) continue;
                    output.write(buffer, 0, count);
                    received += count;
                    progress.update(Math.min(1, (double) received / model.sizeBytes));
                }
            } finally {
                connection.disconnect();
            }
            if (partial.length() != model.sizeBytes) {
                throw new IllegalStateException(String.format(Locale.US,
                        "모델 크기 불일치: %d != %d", partial.length(), model.sizeBytes));
            }
            verify(partial, model.sha256);
            if (!partial.renameTo(complete)) throw new IllegalStateException("모델 파일 확정 실패");
            return complete;
        }

        private static void verify(File file, String expected) throws Exception {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) >= 0) if (count > 0) digest.update(buffer, 0, count);
            }
            StringBuilder actual = new StringBuilder();
            for (byte value : digest.digest()) actual.append(String.format("%02x", value));
            if (!actual.toString().equalsIgnoreCase(expected)) throw new SecurityException("SHA-256 불일치");
        }
    }
}
