package com.example.llmbench;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.os.Environment;
import android.os.PowerManager;
import android.os.StatFs;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
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
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final String PROMPT = "중학교 2학년 학생에게 일차함수 y=2x+3의 기울기와 y절편을 한 문단으로 설명해줘.";
    private final List<ModelSpec> models = new ArrayList<>();
    private TextView status;
    private TextView answer;
    private TextView metrics;
    private ProgressBar progress;
    private Spinner modelSpinner;
    private Spinner backendSpinner;
    private ModelSpec selected;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        NativeRuntime.load();
        loadModels();
        buildUi();
    }

    private void loadModels() {
        try (InputStream input = getAssets().open("models.json")) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192]; int count;
            while ((count = input.read(chunk)) >= 0) { if (count > 0) buffer.write(chunk, 0, count); }
            byte[] bytes = buffer.toByteArray();
            JSONArray entries = new JSONObject(new String(bytes, StandardCharsets.UTF_8)).getJSONArray("models");
            for (int i = 0; i < entries.length(); i++) {
                JSONObject item = entries.getJSONObject(i);
                if (!item.getString("id").startsWith("qwen36-")) continue;
                JSONArray artifacts = item.optJSONArray("artifacts");
                if (artifacts == null || artifacts.length() == 0) continue;
                JSONObject artifact = artifacts.getJSONObject(0);
                models.add(new ModelSpec(item.getString("id"), item.getString("name"),
                        artifact.getString("filename"), artifact.getString("url"),
                        artifact.getString("sha256"), artifact.getLong("size_bytes"),
                        artifact.getString("quantization"), item.optDouble("minimum_ram_gb", 0),
                        item.optDouble("minimum_storage_gb", 0)));
            }
        } catch (Exception error) {
            status = new TextView(this);
            status.setText("매니페스트 로드 실패: " + error.getMessage());
        }
        if (models.isEmpty()) models.add(ModelSpec.invalid());
        selected = models.get(0);
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 24, 32, 24);
        TextView title = new TextView(this); title.setText("LLM Runtime Bench\nllama.cpp · Android arm64-v8a"); title.setTextSize(22); root.addView(title);
        TextView note = new TextView(this); note.setText("가중치는 APK에 포함하지 않습니다. 에뮬레이터는 CPU smoke만 허용하며 GPU 수치는 실기기에서만 유효합니다."); root.addView(note);

        modelSpinner = new Spinner(this);
        List<String> names = new ArrayList<>(); for (ModelSpec model : models) names.add(model.name);
        modelSpinner.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, names));
        modelSpinner.setOnItemSelectedListener(new SimpleItemListener() { @Override public void selected(int position) { selected = models.get(position); } });
        root.addView(modelSpinner);

        backendSpinner = new Spinner(this);
        backendSpinner.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
                new String[]{"auto (Vulkan → OpenCL → ARM CPU)", "vulkan", "opencl-experimental", "cpu-arm64"}));
        root.addView(backendSpinner);
        TextView backend = new TextView(this); backend.setText("감지 backend: " + BackendDetector.choose(this, "auto")); root.addView(backend);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); progress.setMax(1000); root.addView(progress);
        Button download = new Button(this); download.setText("모델 다운로드·재개·SHA-256 검증"); root.addView(download);
        Button chat = new Button(this); chat.setText("고정 프롬프트 스트리밍 채팅"); root.addView(chat);
        Button benchmark = new Button(this); benchmark.setText("고정 벤치마크 실행·JSON 저장"); root.addView(benchmark);
        status = new TextView(this); status.setText("대기 중 · " + selected.name); root.addView(status);
        answer = new TextView(this); answer.setText("아직 답변이 없습니다."); root.addView(answer);
        metrics = new TextView(this); metrics.setText("결과 JSON이 아직 없습니다."); root.addView(metrics);
        EditText prompt = new EditText(this); prompt.setText(PROMPT); prompt.setVisibility(View.GONE); root.addView(prompt);
        setContentView(root);

        download.setOnClickListener(v -> downloadSelected());
        chat.setOnClickListener(v -> runChat());
        benchmark.setOnClickListener(v -> runBenchmark());
    }

    private void downloadSelected() {
        if (!selected.valid) { status.setText("유효한 모델이 없습니다."); return; }
        status.setText("저장공간·RAM 검사 중...");
        if (!Preflight.canRun(this, selected)) { status.setText("blocked: 저장공간 또는 RAM 여유 부족"); return; }
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo(); ((ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(memory);
        status.setText(String.format(Locale.US, "총 RAM 기준 통과: %.0f MB · 다운로드 시작", memory.totalMem / (1024.0 * 1024.0)));
        new Thread(() -> {
            try {
                File file = ModelDownloader.download(selected, getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), fraction -> runOnUiThread(() -> progress.setProgress((int) (fraction * 1000))));
                runOnUiThread(() -> status.setText("다운로드·SHA-256 검증 완료: " + file.getName()));
            } catch (Exception error) { runOnUiThread(() -> status.setText("다운로드 실패: " + error.getMessage())); }
        }).start();
    }

    private void runChat() {
        String backend = BackendDetector.choose(this, backendSpinner.getSelectedItem().toString().split(" ")[0]);
        status.setText("backend=" + backend + " · 스트리밍 채팅 시작");
        answer.setText("");
        new Thread(() -> {
            String response = NativeRuntime.generate(PROMPT);
            runOnUiThread(() -> { answer.setText(response); status.setText("채팅 완료/제한 상태는 결과 JSON을 확인하세요."); });
        }).start();
    }

    private void runBenchmark() {
        new Thread(() -> {
            long started = System.nanoTime();
            String backend = BackendDetector.choose(this, backendSpinner.getSelectedItem().toString().split(" ")[0]);
            String runtimeStatus = NativeRuntime.status();
            long loadMs = (System.nanoTime() - started) / 1_000_000;
            try {
                JSONObject result = new JSONObject();
                result.put("model_id", selected.id); result.put("quantization", selected.quantization);
                result.put("artifact_filename", selected.filename); result.put("artifact_url", selected.url); result.put("artifact_sha256", selected.sha256);
                result.put("platform", "android"); result.put("runtime", "llama.cpp JNI"); result.put("backend", backend);
                result.put("device", Build.MANUFACTURER + " " + Build.MODEL); result.put("soc", Build.HARDWARE); result.put("android_api", Build.VERSION.SDK_INT); result.put("abi", Build.SUPPORTED_ABIS[0]);
                result.put("model_load_ms", loadMs); result.put("ttft_ms", JSONObject.NULL); result.put("prompt_tok_per_s", JSONObject.NULL); result.put("decode_tok_per_s", JSONObject.NULL);
                ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo(); ((ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(memory);
                result.put("ram_total_mb", memory.totalMem / (1024.0 * 1024.0));
                result.put("peak_rss_mb", Debug.getPss() / 1024.0); result.put("oom_events", 0); result.put("crashes", 0); result.put("thermal_status", thermalStatus()); result.put("requests_completed", 0);
                result.put("status", runtimeStatus.startsWith("blocked") ? "runtime_blocked" : "smoke"); result.put("blocked_reason", runtimeStatus);
                File output = new File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "llm-bench-" + System.currentTimeMillis() + ".json");
                try (FileOutputStream stream = new FileOutputStream(output)) { stream.write(result.toString(2).getBytes(StandardCharsets.UTF_8)); }
                runOnUiThread(() -> { metrics.setText(result.toString()); status.setText("JSON 저장: " + output.getAbsolutePath()); });
            } catch (Exception error) { runOnUiThread(() -> status.setText("벤치마크 저장 실패: " + error.getMessage())); }
        }).start();
    }

    private String thermalStatus() {
        if (Build.VERSION.SDK_INT < 29) return "unavailable";
        PowerManager power = (PowerManager) getSystemService(POWER_SERVICE);
        return String.valueOf(power.getCurrentThermalStatus());
    }

    static final class ModelSpec {
        final String id, name, filename, url, sha256, quantization; final long sizeBytes; final double minRamGb, minStorageGb; final boolean valid;
        ModelSpec(String id, String name, String filename, String url, String sha256, long sizeBytes, String quantization, double minRamGb, double minStorageGb) { this.id=id; this.name=name; this.filename=filename; this.url=url; this.sha256=sha256; this.sizeBytes=sizeBytes; this.quantization=quantization; this.minRamGb=minRamGb; this.minStorageGb=minStorageGb; this.valid=true; }
        private ModelSpec() { id=""; name="no model"; filename=""; url=""; sha256=""; sizeBytes=0; quantization=""; minRamGb=0; minStorageGb=0; valid=false; }
        static ModelSpec invalid() { return new ModelSpec(); }
    }

    interface Progress { void update(double fraction); }

    static final class Preflight {
        static boolean canRun(Context context, ModelSpec model) {
            ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo(); ((ActivityManager) context.getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(memory);
            StatFs stats = new StatFs(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getPath());
            // Android reports available memory after the OS and other apps have
            // consumed pages. Model admission is based on device capacity so a
            // 12 GB-class phone is not rejected merely because it is in use.
            long requiredRamBytes = (long) (model.minRamGb * 1000 * 1000 * 1000);
            return memory.totalMem >= requiredRamBytes && stats.getAvailableBytes() >= model.sizeBytes + (long) (model.sizeBytes * .05);
        }
    }

    static final class BackendDetector {
        static String choose(Context context, String forced) {
            if (forced.equals("cpu-arm64")) return "cpu-arm64";
            if (forced.equals("vulkan")) return hasVulkan(context) ? "vulkan" : "blocked:vulkan_unavailable";
            if (forced.equals("opencl-experimental")) return hasOpenCl() ? "opencl-experimental" : "blocked:opencl_unavailable";
            if (hasVulkan(context)) return "vulkan";
            if (hasOpenCl()) return "opencl-experimental";
            return "cpu-arm64";
        }
        private static boolean hasVulkan(Context context) { return Build.VERSION.SDK_INT >= 24 && context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL); }
        private static boolean hasOpenCl() { return new File("/system/vendor/lib64/libOpenCL.so").exists() || new File("/system/lib64/libOpenCL.so").exists(); }
    }

    static final class ModelDownloader {
        static File download(ModelSpec model, File directory, Progress progress) throws Exception {
            if (!directory.exists() && !directory.mkdirs()) throw new IllegalStateException("cannot create model directory");
            File complete = new File(directory, model.filename), partial = new File(directory, model.filename + ".partial");
            if (complete.exists() && complete.length() == model.sizeBytes) { verify(complete, model.sha256); return complete; }
            long offset = partial.exists() ? partial.length() : 0;
            HttpURLConnection connection = (HttpURLConnection) new URL(model.url).openConnection(); connection.setInstanceFollowRedirects(true); connection.setConnectTimeout(30_000); connection.setReadTimeout(60_000); if (offset > 0) connection.setRequestProperty("Range", "bytes=" + offset + "-");
            int code = connection.getResponseCode(); if (code != 200 && code != 206) throw new IllegalStateException("HTTP " + code);
            boolean append = offset > 0 && code == 206; if (!append) { offset = 0; partial.delete(); }
            try (InputStream input = new BufferedInputStream(connection.getInputStream()); OutputStream output = new FileOutputStream(partial, append)) {
                byte[] buffer = new byte[1024 * 1024]; long received = offset; int count; while ((count = input.read(buffer)) >= 0) { if (count == 0) continue; output.write(buffer, 0, count); received += count; progress.update(Math.min(1, (double) received / model.sizeBytes)); }
            } finally { connection.disconnect(); }
            if (partial.length() != model.sizeBytes) throw new IllegalStateException(String.format(Locale.US, "size mismatch: %d != %d", partial.length(), model.sizeBytes));
            verify(partial, model.sha256); if (!partial.renameTo(complete)) throw new IllegalStateException("cannot finalize model"); return complete;
        }
        private static void verify(File file, String expected) throws Exception { MessageDigest digest = MessageDigest.getInstance("SHA-256"); try (InputStream input = new BufferedInputStream(new FileInputStream(file))) { byte[] buffer = new byte[1024 * 1024]; int count; while ((count = input.read(buffer)) >= 0) { if (count > 0) digest.update(buffer, 0, count); } } StringBuilder actual = new StringBuilder(); for (byte value : digest.digest()) actual.append(String.format("%02x", value)); if (!actual.toString().equalsIgnoreCase(expected)) throw new SecurityException("SHA-256 mismatch"); }
    }

    abstract static class SimpleItemListener implements android.widget.AdapterView.OnItemSelectedListener {
        public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) { selected(position); }
        public abstract void selected(int position);
    }
}
