param(
    [string]$Destination = "C:\llm-cache\browser-vlm"
)

$ErrorActionPreference = "Stop"
$files = @(
    @{
        Url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/0df8719db7180cedababc2bc589abfe5e8ebcd1f/LFM2.5-VL-1.6B-Q4_K_M.gguf"
        Name = "LFM2.5-VL-1.6B-Q4_K_M.gguf"
        Size = 730896256
        Sha256 = "aefc3c97c9eb30d9c0dd6af4c38250f5f5106b57c8cf92de7914c7d0a9c94da2"
    },
    @{
        Url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/0df8719db7180cedababc2bc589abfe5e8ebcd1f/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"
        Name = "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"
        Size = 583109888
        Sha256 = "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d"
    },
    @{
        Url = "https://huggingface.co/mradermacher/Qwen3.5-4B-GGUF/resolve/1a5df2c0cba51dae8ac5888420360d8703707171/Qwen3.5-4B.Q4_K_M.gguf"
        Name = "Qwen3.5-4B.Q4_K_M.gguf"
        Size = 2708804800
        Sha256 = "51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39"
    },
    @{
        Url = "https://huggingface.co/mradermacher/Qwen3.5-4B-GGUF/resolve/1a5df2c0cba51dae8ac5888420360d8703707171/Qwen3.5-4B.mmproj-Q8_0.gguf"
        Name = "Qwen3.5-4B.mmproj-Q8_0.gguf"
        Size = 366894656
        Sha256 = "40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad"
    },
    @{
        Url = "https://huggingface.co/mradermacher/Qwen3.5-2B-GGUF/resolve/33cc6944e40cb93e38332cd46b2a6e3c6acf081e/Qwen3.5-2B.Q4_K_M.gguf"
        Name = "Qwen3.5-2B.Q4_K_M.gguf"
        Size = 1270808896
        Sha256 = "d772079a853f3494be962e1bde20b4dbf1454c89d1da4c686cf701de19fc73f1"
    },
    @{
        Url = "https://huggingface.co/mradermacher/Qwen3.5-2B-GGUF/resolve/33cc6944e40cb93e38332cd46b2a6e3c6acf081e/Qwen3.5-2B.mmproj-Q8_0.gguf"
        Name = "Qwen3.5-2B.mmproj-Q8_0.gguf"
        Size = 364664384
        Sha256 = "526dbf85f350baf3a5107b1f14e629e94571c7cbab4277476fbdaaa8c4a31a64"
    }
)

$root = New-Item -ItemType Directory -Path $Destination -Force
foreach ($file in $files) {
    $target = Join-Path $root.FullName $file.Name
    if ((Test-Path -LiteralPath $target) -and (Get-Item -LiteralPath $target).Length -eq $file.Size) {
        $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $file.Sha256) {
            Write-Host "verified: $($file.Name)"
            continue
        }
    }
    Write-Host "downloading: $($file.Name)"
    & curl.exe --fail --location --retry 5 --retry-delay 2 --output "$target.partial" $file.Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $($file.Name)" }
    $download = Get-Item -LiteralPath "$target.partial"
    if ($download.Length -ne $file.Size) { throw "size mismatch for $($file.Name): $($download.Length)" }
    $actual = (Get-FileHash -LiteralPath $download.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $file.Sha256) { throw "SHA-256 mismatch for $($file.Name): $actual" }
    Move-Item -LiteralPath $download.FullName -Destination $target -Force
    Write-Host "verified: $($file.Name)"
}
