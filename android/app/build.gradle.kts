plugins { id("com.android.application") }

android {
    namespace = "com.example.llmbench"
    compileSdk = 35
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "com.example.llmbench"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { cppFlags += "-std=c++17" } }
    }

    buildTypes {
        debug { isMinifyEnabled = false }
        release { isMinifyEnabled = true; proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    sourceSets["main"].assets.srcDir(rootProject.file("../configs"))
}
