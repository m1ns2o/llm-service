#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

namespace {
constexpr const char * TAG = "LlmNativeRuntime";
std::mutex runtime_mutex;
llama_model * model = nullptr;
std::string active_model_path;
std::string active_backend = "unloaded";
std::string active_requested_backend = "unloaded";
int active_threads = 1;
int active_context_size = 4096;
bool backend_initialized = false;
std::atomic<bool> interrupt_requested{false};

void ensure_backends_initialized() {
    if (!backend_initialized) {
        llama_backend_init();
        ggml_backend_load_all();
        backend_initialized = true;
    }
}

std::string from_jstring(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const jchar * chars = env->GetStringChars(value, nullptr);
    if (chars == nullptr) return {};
    const jsize length = env->GetStringLength(value);
    std::string result;
    result.reserve(static_cast<size_t>(length) * 3);
    for (jsize i = 0; i < length; ++i) {
        uint32_t codepoint = chars[i];
        if (codepoint >= 0xD800 && codepoint <= 0xDBFF && i + 1 < length) {
            uint32_t low = chars[i + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) {
                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                ++i;
            }
        }
        if (codepoint <= 0x7F) result.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7FF) {
            result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }
    env->ReleaseStringChars(value, chars);
    return result;
}

jstring to_jstring(JNIEnv * env, const std::string & value) {
    std::vector<jchar> utf16;
    utf16.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        uint32_t codepoint;
        unsigned char first = static_cast<unsigned char>(value[i++]);
        if (first < 0x80) codepoint = first;
        else if ((first & 0xE0) == 0xC0 && i < value.size()) {
            codepoint = ((first & 0x1F) << 6) | (static_cast<unsigned char>(value[i++]) & 0x3F);
        } else if ((first & 0xF0) == 0xE0 && i + 1 < value.size()) {
            const uint32_t second = static_cast<unsigned char>(value[i++]);
            const uint32_t third = static_cast<unsigned char>(value[i++]);
            codepoint = ((first & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F);
        } else if ((first & 0xF8) == 0xF0 && i + 2 < value.size()) {
            const uint32_t second = static_cast<unsigned char>(value[i++]);
            const uint32_t third = static_cast<unsigned char>(value[i++]);
            const uint32_t fourth = static_cast<unsigned char>(value[i++]);
            codepoint = ((first & 0x07) << 18) | ((second & 0x3F) << 12) |
                ((third & 0x3F) << 6) | (fourth & 0x3F);
        } else {
            codepoint = 0xFFFD;
        }
        if (codepoint <= 0xFFFF) utf16.push_back(static_cast<jchar>(codepoint));
        else {
            codepoint -= 0x10000;
            utf16.push_back(static_cast<jchar>(0xD800 + (codepoint >> 10)));
            utf16.push_back(static_cast<jchar>(0xDC00 + (codepoint & 0x3FF)));
        }
    }
    return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

void release_model() {
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    active_model_path.clear();
    active_backend = "unloaded";
    active_requested_backend = "unloaded";
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

ggml_backend_dev_t find_backend_device(const std::string & backend) {
    const std::string needle = lowercase(backend);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * device_name = ggml_backend_dev_name(device);
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const char * registry_name = registry == nullptr ? nullptr : ggml_backend_reg_name(registry);
        const std::string searchable = lowercase(
            std::string(device_name == nullptr ? "" : device_name) + " " +
            std::string(registry_name == nullptr ? "" : registry_name));
        if (searchable.find(needle) != std::string::npos) return device;
    }
    return nullptr;
}

std::vector<std::string> parse_backend_order(const std::string & requested) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= requested.size()) {
        const size_t end = requested.find(',', start);
        const std::string candidate = lowercase(requested.substr(
            start, end == std::string::npos ? std::string::npos : end - start));
        if ((candidate == "opencl" || candidate == "vulkan" || candidate == "cpu-arm64") &&
            std::find(result.begin(), result.end(), candidate) == result.end()) {
            result.push_back(candidate);
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (std::find(result.begin(), result.end(), "cpu-arm64") == result.end()) {
        result.push_back("cpu-arm64");
    }
    return result;
}

std::string available_backends() {
    ensure_backends_initialized();
    std::vector<std::string> result;
    for (const std::string & candidate : {"opencl", "vulkan"}) {
        if (find_backend_device(candidate) != nullptr) result.push_back(candidate);
    }
    result.push_back("cpu-arm64");
    std::string joined;
    for (const std::string & candidate : result) {
        if (!joined.empty()) joined += ',';
        joined += candidate;
    }
    return joined;
}

std::string apply_chat_template(
    const std::vector<std::string> & roles,
    const std::vector<std::string> & contents) {
    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }
    const char * chat_template = llama_model_chat_template(model, nullptr);
    int32_t required = llama_chat_apply_template(
        chat_template, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0) return contents.empty() ? std::string() : contents.back();
    std::vector<char> buffer(static_cast<size_t>(required) + 1, '\0');
    int32_t written = llama_chat_apply_template(
        chat_template, messages.data(), messages.size(), true, buffer.data(), static_cast<int32_t>(buffer.size()));
    return written > 0 ? std::string(buffer.data(), static_cast<size_t>(written))
                       : (contents.empty() ? std::string() : contents.back());
}

bool probe_model_backend(const std::string & backend) {
    try {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 512;
        params.n_batch = 32;
        params.n_ubatch = 32;
        params.n_threads = active_threads;
        params.n_threads_batch = active_threads;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const bool gpu_backend = backend == "vulkan" || backend == "opencl";
        params.offload_kqv = gpu_backend;
        params.op_offload = gpu_backend;
        params.no_perf = false;

        llama_context * context = llama_init_from_model(model, params);
        if (context == nullptr) return false;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        constexpr const char * probe_prompt = "Hello";
        int count = -llama_tokenize(vocab, probe_prompt, 5, nullptr, 0, true, true);
        if (count <= 0) {
            llama_free(context);
            return false;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(count));
        if (llama_tokenize(vocab, probe_prompt, 5, tokens.data(), tokens.size(), true, true) < 0) {
            llama_free(context);
            return false;
        }
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        const bool ready = llama_decode(context, batch) == 0;
        llama_free(context);
        return ready;
    } catch (const std::exception & error) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "%s warmup failed: %s",
                            backend.c_str(), error.what());
        return false;
    } catch (...) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "%s warmup failed with unknown error",
                            backend.c_str());
        return false;
    }
}

std::string generate_text(
    JNIEnv * env,
    jobject callback,
    const std::vector<std::string> & roles,
    const std::vector<std::string> & contents,
    int max_tokens,
    float temperature) {
    if (model == nullptr) return "실행 차단: 먼저 모델을 로드하세요.";
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = apply_chat_template(roles, contents);
    int token_count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (token_count <= 0) return "실행 실패: 프롬프트 토큰화 오류";

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                       prompt_tokens.size(), true, true) < 0) {
        return "실행 실패: 프롬프트 토큰화 오류";
    }

    max_tokens = std::clamp(max_tokens, 1, 2048);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(
        static_cast<uint32_t>(active_context_size),
        static_cast<uint32_t>(prompt_tokens.size() + max_tokens + 32));
    context_params.n_batch = std::min<uint32_t>(context_params.n_ctx,
        std::max<uint32_t>(512, static_cast<uint32_t>(prompt_tokens.size())));
    context_params.n_ubatch = std::min<uint32_t>(256, context_params.n_batch);
    context_params.n_threads = active_threads;
    context_params.n_threads_batch = active_threads;
    // Adreno 830's Android Vulkan driver rejects llama.cpp's Flash Attention
    // compute pipeline with VK_ERROR_UNKNOWN. Keep the portable Vulkan
    // attention path; matrix/vector operations are still offloaded to GPU.
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const bool gpu_backend = active_backend == "vulkan" || active_backend == "opencl";
    context_params.offload_kqv = gpu_backend;
    context_params.op_offload = gpu_backend;
    context_params.no_perf = false;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) return "실행 실패: 컨텍스트 메모리를 확보할 수 없습니다.";

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    std::string output;
    output.reserve(static_cast<size_t>(max_tokens) * 4);
    jmethodID on_token = callback == nullptr ? nullptr
        : env->GetMethodID(env->GetObjectClass(callback), "onToken", "(Ljava/lang/String;)V");
    interrupt_requested.store(false, std::memory_order_release);
    int generated = 0;
    while (generated < max_tokens && !interrupt_requested.load(std::memory_order_acquire)) {
        if (llama_decode(context, batch) != 0) {
            if (generated == 0) output = "실행 실패: 백엔드 첫 디코드 오류";
            else output += "\n\n[생성 중 llama_decode 오류]";
            break;
        }
        llama_token token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        int piece_size = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
        if (piece_size < 0) {
            std::vector<char> piece(static_cast<size_t>(-piece_size));
            piece_size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            if (piece_size > 0) {
                std::string token_piece(piece.data(), static_cast<size_t>(piece_size));
                output.append(token_piece);
                if (on_token != nullptr) {
                    jstring token_value = to_jstring(env, token_piece);
                    env->CallVoidMethod(callback, on_token, token_value);
                    env->DeleteLocalRef(token_value);
                    if (env->ExceptionCheck()) {
                        env->ExceptionClear();
                        interrupt_requested.store(true, std::memory_order_release);
                    }
                }
            }
        }
        batch = llama_batch_get_one(&token, 1);
        generated++;
    }

    __android_log_print(ANDROID_LOG_INFO, TAG,
                        "generation backend=%s prompt_tokens=%d generated_tokens=%d",
                        active_backend.c_str(), token_count, generated);
    llama_sampler_free(sampler);
    llama_free(context);
    return output.empty() ? "[빈 응답]" : output;
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_statusNative(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    std::string status = "ready: llama.cpp ";
    status += llama_version();
    status += model == nullptr ? " · model=unloaded" : " · model=loaded · backend=" + active_backend;
    return to_jstring(env, status);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_availableBackendsNative(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    return to_jstring(env, available_backends());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_loadModelNative(
    JNIEnv * env, jclass, jstring path_value, jstring backend_value, jint threads, jint context_size) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    ensure_backends_initialized();

    const std::string path = from_jstring(env, path_value);
    const std::string requested_backend = from_jstring(env, backend_value);
    if (path.empty()) return to_jstring(env, "blocked: model path is empty");
    if (path == active_model_path && model != nullptr && requested_backend == active_requested_backend) {
        return to_jstring(env, "ready: model already loaded · backend=" + active_backend);
    }

    release_model();
    active_threads = std::clamp<int>(threads, 1, 8);
    active_context_size = std::clamp<int>(context_size, 512, 8192);
    for (const std::string & candidate : parse_backend_order(requested_backend)) {
        llama_model_params params = llama_model_default_params();
        const bool wants_gpu = candidate == "vulkan" || candidate == "opencl";
        std::vector<ggml_backend_dev_t> selected_devices;
        if (wants_gpu) {
            ggml_backend_dev_t device = find_backend_device(candidate);
            if (device == nullptr) {
                __android_log_print(ANDROID_LOG_WARN, TAG,
                                    "%s backend is unavailable; trying fallback", candidate.c_str());
                continue;
            }
            selected_devices = {device, nullptr};
            params.devices = selected_devices.data();
            __android_log_print(ANDROID_LOG_INFO, TAG, "selecting %s device=%s description=%s",
                                candidate.c_str(), ggml_backend_dev_name(device),
                                ggml_backend_dev_description(device));
        }
        params.n_gpu_layers = wants_gpu ? -1 : 0;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        params.main_gpu = 0;
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
        params.use_extra_bufts = true;
        model = llama_model_load_from_file(path.c_str(), params);
        if (model != nullptr && probe_model_backend(candidate)) {
            active_backend = candidate;
            active_requested_backend = requested_backend;
            break;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        __android_log_print(ANDROID_LOG_WARN, TAG,
                            "%s load or warmup failed; trying fallback", candidate.c_str());
    }
    if (model == nullptr) {
        release_model();
        return to_jstring(env, "blocked: GGUF model load failed");
    }

    active_model_path = path;
    return to_jstring(env, "ready: model loaded · backend=" + active_backend +
        " · threads=" + std::to_string(active_threads));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmbench_NativeRuntime_generateNative(
    JNIEnv * env, jclass, jobjectArray role_values, jobjectArray content_values,
    jint max_tokens, jfloat temperature, jobject callback) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    const jsize role_count = role_values == nullptr ? 0 : env->GetArrayLength(role_values);
    const jsize content_count = content_values == nullptr ? 0 : env->GetArrayLength(content_values);
    if (role_count == 0 || role_count != content_count) {
        return to_jstring(env, "실행 실패: 잘못된 대화 메시지");
    }
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(role_count);
    contents.reserve(content_count);
    for (jsize i = 0; i < role_count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(role_values, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(content_values, i));
        roles.push_back(from_jstring(env, role));
        contents.push_back(from_jstring(env, content));
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }
    return to_jstring(env, generate_text(env, callback, roles, contents, max_tokens, temperature));
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llmbench_NativeRuntime_requestInterruptNative(JNIEnv *, jclass) {
    interrupt_requested.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_llmbench_NativeRuntime_unloadNative(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    release_model();
}
