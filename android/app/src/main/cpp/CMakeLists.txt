cmake_minimum_required(VERSION 3.22.1)
project(llama_runtime_stub LANGUAGES CXX)

# This target deliberately has no model weights. Replace this source with the
# output of scripts/build_llama_runtime.sh when the pinned llama.cpp source and
# Android backend toolchains are available.
add_library(llama_runtime SHARED llama_runtime_stub.cpp)
# Android API 35+ devices may use 16 KB pages. Keep the JNI shared object
# loadable on those devices; the same flag must be retained by the eventual
# llama.cpp native link target.
target_link_options(llama_runtime PRIVATE "-Wl,-z,max-page-size=16384")
find_library(log-lib log)
target_link_libraries(llama_runtime ${log-lib})
