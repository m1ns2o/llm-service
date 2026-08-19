cmake_minimum_required(VERSION 3.22.1)
project(llama_runtime LANGUAGES C CXX)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/llama.cpp"
    CACHE PATH "Pinned llama.cpp checkout")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp is required. Run scripts/prepare_android_runtime.ps1 before building the APK.")
endif()

set(SPIRV_HEADERS_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/SPIRV-Headers/install/share/cmake/SPIRV-Headers")
if(NOT EXISTS "${SPIRV_HEADERS_CONFIG}/SPIRV-HeadersConfig.cmake")
    message(FATAL_ERROR
        "SPIRV-Headers is required. Run scripts/prepare_android_runtime.ps1 before building the APK.")
endif()
set(SPIRV-Headers_DIR "${SPIRV_HEADERS_CONFIG}" CACHE PATH "" FORCE)

set(VULKAN_HEADERS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/Vulkan-Headers/include")
if(NOT EXISTS "${VULKAN_HEADERS_DIR}/vulkan/vulkan.hpp")
    message(FATAL_ERROR
        "Vulkan-Headers is required. Run scripts/prepare_android_runtime.ps1 before building the APK.")
endif()
# Keep the NDK Vulkan loader/library selected by the Android toolchain while
# supplying the full Khronos C++ headers required by ggml-vulkan.
set(Vulkan_INCLUDE_DIR "${VULKAN_HEADERS_DIR}" CACHE PATH "" FORCE)

# The NDK ships a host glslc binary, but FindVulkan does not discover it when
# CMake is operating through the Android cross-compilation toolchain.
set(ANDROID_GLSLC "${ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe")
if(CMAKE_HOST_WIN32 AND EXISTS "${ANDROID_GLSLC}")
    set(Vulkan_GLSLC_EXECUTABLE "${ANDROID_GLSLC}" CACHE FILEPATH "" FORCE)
endif()

# llama.cpp needs to compile its Vulkan shader generator for the Windows host
# while the actual runtime is cross-compiled by the Android NDK. Android
# Studio/Gradle does not normally inherit a Visual Studio developer shell, so
# import the MSVC host environment explicitly before llama.cpp probes PATH.
if(CMAKE_HOST_WIN32 AND CMAKE_CROSSCOMPILING)
    set(VSWHERE "$ENV{SystemDrive}/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if(EXISTS "${VSWHERE}")
        execute_process(
            COMMAND "${VSWHERE}" -latest -products *
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
                    -property installationPath
            OUTPUT_VARIABLE VS_INSTALLATION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(VSDEVCMD "${VS_INSTALLATION}/Common7/Tools/VsDevCmd.bat")
        if(EXISTS "${VSDEVCMD}")
            execute_process(
                COMMAND cmd.exe /d /c call "${VSDEVCMD}"
                        -arch=x64 -host_arch=x64 && set
                OUTPUT_VARIABLE VS_ENV
                RESULT_VARIABLE VS_ENV_RESULT)
            if(VS_ENV_RESULT EQUAL 0)
                foreach(VS_ENV_NAME IN ITEMS INCLUDE LIB LIBPATH)
                    string(REGEX MATCH
                        "(^|[\r\n])${VS_ENV_NAME}=([^\r\n]*)"
                        VS_ENV_MATCH "${VS_ENV}")
                    if(VS_ENV_MATCH)
                        set(ENV{${VS_ENV_NAME}} "${CMAKE_MATCH_2}")
                    endif()
                endforeach()
                # Windows emits this particular variable as `Path`, while
                # CMake's environment lookup below uses the canonical `PATH`.
                string(REGEX MATCH "(^|[\r\n])Path=([^\r\n]*)"
                    VS_PATH_MATCH "${VS_ENV}")
                if(VS_PATH_MATCH)
                    set(ENV{PATH} "${CMAKE_MATCH_2}")
                endif()
                unset(HOST_C_COMPILER CACHE)
                unset(HOST_CXX_COMPILER CACHE)
            else()
                message(FATAL_ERROR "Could not initialize the MSVC host build environment")
            endif()
        endif()
    endif()
endif()

# llama.cpp cross-compiles a small host-side Vulkan shader generator through
# ExternalProject. Ensure that nested host CMake can resolve the same Ninja
# executable selected by Android Gradle Plugin.
get_filename_component(ANDROID_CMAKE_BIN_DIR "${CMAKE_MAKE_PROGRAM}" DIRECTORY)
set(ENV{PATH} "${ANDROID_CMAKE_BIN_DIR};$ENV{PATH}")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# NDK r26's prebuilt libomp is not 16 KB page-aligned. ggml's own ARM thread
# pool remains enabled and avoids shipping that incompatible shared library.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
else()
    # KleidiAI is ARM-specific; x86_64 is used only by emulator debug builds.
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
endif()
set(GGML_CPU_REPACK ON CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(OPENCL_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/OpenCL-ICD-Loader")
    set(OpenCL_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/OpenCL-Headers" CACHE PATH "" FORCE)
    set(OpenCL_LIBRARY
        "${OPENCL_ROOT}/build-android2/libOpenCL.so" CACHE FILEPATH "" FORCE)
    if(NOT EXISTS "${OpenCL_LIBRARY}")
        message(FATAL_ERROR
            "Android OpenCL loader is required. Run scripts/prepare_android_runtime.ps1 first.")
    endif()
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
else()
    set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "" FORCE)
endif()
# FindPython follows the Android root path during cross compilation. Resolve
# the host interpreter through the Windows launcher for kernel embedding.
if(CMAKE_HOST_WIN32)
    set(ENV{PYTHONUTF8} "1")
    find_program(HOST_PY_LAUNCHER py REQUIRED NO_CMAKE_FIND_ROOT_PATH)
    execute_process(
        COMMAND "${HOST_PY_LAUNCHER}" -3 -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE HOST_PYTHON
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    set(Python3_EXECUTABLE "${HOST_PYTHON}" CACHE FILEPATH "" FORCE)
endif()
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# ggml-vulkan includes the SPIR-V C++ definitions directly in addition to
# linking the CMake package used by its host shader generator.
if(TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/SPIRV-Headers/include")
endif()

add_library(llama_runtime SHARED llama_runtime_jni.cpp)
target_compile_features(llama_runtime PRIVATE cxx_std_17)
target_include_directories(llama_runtime PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include")
# Android API 35+ devices may use 16 KB pages. Keep the JNI shared object
# loadable on those devices; the same flag must be retained by the eventual
# llama.cpp native link target.
target_link_options(llama_runtime PRIVATE "-Wl,-z,max-page-size=16384")
find_library(log-lib log)
find_library(android-lib android)
target_link_libraries(llama_runtime PRIVATE llama ggml ${log-lib} ${android-lib})
