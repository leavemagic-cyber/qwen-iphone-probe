# qwen-iphone-probe

Minimal open-source (MIT) iOS XCTest probe to measure whether the official **Qwen3-VL-2B-Instruct** GGUF + vision projector (`mmproj`) loads and runs on a physical iPhone via **llama.cpp** (`mtmd` + Metal).

## What It Does

A GitHub Actions macOS workflow builds an ad-hoc signed **Release** Firebase Test Lab XCTest archive (`qwen-iphone-probe-xctest.zip`) containing:
- `<scheme>.xctestrun` at the zip root
- `Release-iphoneos/QwenProbeHost.app` (embedding `llama.framework` from a pinned `ggml-org/llama.cpp` release)
- `Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest`

At test runtime (`QwenProbeTests`), five XCTest methods run in alphabetical order so a multi-test harness can isolate settings even across process restarts:
- All four methods keep the A13 crash fix: `output.weight` / tied `token_embd.weight` on `ggml_backend_cpu_buffer_type()` and `cparams.op_offload = false`, with every transformer layer on Metal.
- `test1_clipCpu448`: vision encoder (CLIP) on the CPU with every online core, 448x448 image.
- `test2_clipCpu768`: CLIP on the CPU with every online core, 576x768 image.
- `test3_clipGpu768`: CLIP on Metal, 576x768 image.
- `test4_clipCpu448Threads2`: CLIP on the CPU with 2 threads, 448x448 image.

Each test method:
1. Downloads the official `Qwen/Qwen3-VL-2B-Instruct-GGUF` weights (`Qwen3VL-2B-Instruct-Q4_K_M.gguf`) and matching vision projector (`mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`) once into the app's `Library/Caches/qwenprobe-models/` directory and reuses cached files across methods.
2. Generates a deterministic `448x448` synthetic RGB test image in memory (a red square centered on a blue background).
3. Loads the language model and vision projector, encodes the image **twice** (reporting both cold `image_encode_seconds` and warm `image_encode_warm_seconds`), logs per-step `phys_footprint_bytes`, and evaluates:
   > `"What colours and shapes are in this image? Answer in one sentence."`
   generating up to 64 tokens.
4. Prints a single-line JSON report prefixed with `QWENPROBE_RESULT ` and attaches it as an `XCTAttachment` containing:
   - `variant`
   - `device_model_identifier`
   - `ios_version`
   - `os_proc_available_memory_before_load_bytes`
   - `phys_footprint_after_model_load_bytes`
   - `peak_phys_footprint_bytes` (`task_vm_info`)
   - `model_load_seconds`
   - `image_encode_seconds`
   - `image_encode_warm_seconds`
   - `time_to_first_token_seconds`
   - `generated_tokens_per_second`
   - `generated_tokens`
   - `generated_text`
   - `error`

`test5_appProfileTenLooks` runs the app profile (vision encoder on Metal, 576x768) ten times in a row and prints a `QWENPROBE_LOOK` line per look with its wall time and the thermal state before and after.

## License

MIT (see [LICENSE](LICENSE)). Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) at pinned release tag `b12020`, and downloads official [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (Apache-2.0) weights at test time.
