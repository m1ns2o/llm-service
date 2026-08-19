plugins { id("com.android.application") }

val webProjectDir = rootProject.file("../web")
val generatedWebDir = rootProject.file("../web/.output/public")
val packagedWebDir = project.file("src/main/assets/web")

val buildWebUi by tasks.registering(Exec::class) {
    workingDir = webProjectDir
    environment("NUXT_ANDROID_BUILD", "1")
    commandLine("npm.cmd", "run", "generate:android")
    inputs.dir(webProjectDir.resolve("app"))
    inputs.file(webProjectDir.resolve("nuxt.config.ts"))
    outputs.dir(generatedWebDir)
}

val syncWebUi by tasks.registering(Copy::class) {
    dependsOn(buildWebUi)
    from(generatedWebDir)
    into(packagedWebDir)
}

android {
    namespace = "com.example.llmbench"
    compileSdk = 35
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "com.example.llmbench"
        // The native Vulkan backend links Vulkan 1.1 core entry points that
        // Android exposes from API 28 onward. Older devices cannot safely use
        // this fully native GPU path.
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        externalNativeBuild { cmake { cppFlags += "-std=c++17" } }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            // Keep an x86_64 runtime in debug APKs so the complete native path
            // can be exercised on the Android tablet emulator.
            ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        }
        release {
            isMinifyEnabled = true
            ndk { abiFilters += "arm64-v8a" }
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    sourceSets["main"].assets.srcDir(rootProject.file("../configs"))
    packaging {
        // libOpenCL.so is a link-time stub only. Supported Android devices
        // provide the vendor implementation declared in the manifest.
        jniLibs.excludes += "**/libOpenCL.so"
    }
}

tasks.named("preBuild").configure { dependsOn(syncWebUi) }

dependencies {
    implementation("androidx.webkit:webkit:1.14.0")
}
