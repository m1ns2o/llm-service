package com.example.llmbench;

import android.app.Activity;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

/** ADB-only debug entry point for repeatable physical-device backend tests. */
public final class NativeSmokeActivity extends Activity {
    private static final String TAG = "NativeSmoke";

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        new Thread(this::runSmoke, "native-smoke").start();
    }

    private void runSmoke() {
        try {
            File allowedRoot = getExternalFilesDir(null);
            File model = new File(getIntent().getStringExtra("model"));
            if (allowedRoot == null || !model.getCanonicalPath().startsWith(allowedRoot.getCanonicalPath() + File.separator)) {
                throw new SecurityException("Model must be inside the app external-files directory");
            }
            String backend = getIntent().getStringExtra("backend");
            if (backend == null) backend = "vulkan";
            int maxTokens = getIntent().getIntExtra("max_tokens", 64);

            NativeRuntime.load();
            Log.i(TAG, "runtime=" + NativeRuntime.status());
            long loadStart = SystemClock.elapsedRealtimeNanos();
            String loadResult = NativeRuntime.loadModel(
                    model.getAbsolutePath(), backend,
                    Math.max(2, Math.min(8, Runtime.getRuntime().availableProcessors() - 1)), 2048);
            double loadSeconds = (SystemClock.elapsedRealtimeNanos() - loadStart) / 1_000_000_000.0;
            Log.i(TAG, "load backend=" + backend + " seconds=" + loadSeconds + " result=" + loadResult);

            AtomicInteger pieces = new AtomicInteger();
            long generationStart = SystemClock.elapsedRealtimeNanos();
            String result = NativeRuntime.generate(
                    new String[] { "user" },
                    new String[] { "한국어로 한 문장만 답해줘. 대한민국의 수도는 어디야?" },
                    maxTokens, 0.1f,
                    token -> {
                        pieces.incrementAndGet();
                        Log.d(TAG, "token=" + token);
                    });
            double generationSeconds = (SystemClock.elapsedRealtimeNanos() - generationStart) / 1_000_000_000.0;
            Log.i(TAG, "generate seconds=" + generationSeconds
                    + " pieces=" + pieces.get()
                    + " pieces_per_second=" + (pieces.get() / generationSeconds)
                    + " result=" + result);
            NativeRuntime.unload();
        } catch (Throwable error) {
            Log.e(TAG, "smoke failed", error);
        } finally {
            runOnUiThread(this::finish);
        }
    }
}
