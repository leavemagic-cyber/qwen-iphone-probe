# qwen-iphone-probe

Minimal open-source (MIT) iOS XCTest probe to measure whether the official **Qwen3-VL-2B-Instruct** GGUF + vision projector (`mmproj`) loads and runs on a physical iPhone via **llama.cpp** (`mtmd` + Metal).

## What It Does

A GitHub Actions macOS workflow builds an ad-hoc signed **Release** Firebase Test Lab XCTest archive (`qwen-iphone-probe-xctest.zip`) containing:
- `<scheme>.xctestrun` at the zip root
- `Release-iphoneos/QwenProbeHost.app` (embedding `llama.framework` from a pinned `ggml-org/llama.cpp` release)
- `Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest`

At test runtime (`QwenProbeTests/testQwenOnDevice`), the probe:
1. Downloads the official `Qwen/Qwen3-VL-2B-Instruct-GGUF` weights (`Qwen3VL-2B-Instruct-Q4_K_M.gguf`) and matching vision projector (`mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`) from Hugging Face directly to temporary disk storage, logging URLs and exact byte sizes.
2. Generates a deterministic `448x448` synthetic RGB test image in memory (a red square centered on a blue background).
3. Loads the language model and vision projector onto Metal (`n_gpu_layers = -1`, `use_gpu = true`) and evaluates:
   > `"What colours and shapes are in this image? Answer in one sentence."`
   generating up to 64 tokens.
4. Prints a single-line JSON report prefixed with `QWENPROBE_RESULT ` and attaches it as an `XCTAttachment` containing:
   - `device_model_identifier`
   - `ios_version`
   - `os_proc_available_memory_before_load_bytes`
   - `peak_phys_footprint_bytes` (`task_vm_info`)
   - `model_load_seconds`
   - `image_encode_seconds`
   - `time_to_first_token_seconds`
   - `generated_tokens_per_second`
   - `generated_text`
   - `error`

## License

MIT (see [LICENSE](LICENSE)). Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) at pinned release tag `b12020`, and downloads official [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (Apache-2.0) weights at test time.
