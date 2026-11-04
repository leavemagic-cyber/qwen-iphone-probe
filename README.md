# qwen-iphone-probe

Minimal open-source (MIT) iOS XCTest probe to measure whether the official **Qwen3-VL-2B-Instruct** GGUF + vision projector (`mmproj`) loads and runs on a physical iPhone via **llama.cpp** (`mtmd` + Metal).

## What It Does

A GitHub Actions macOS workflow builds an ad-hoc signed **Release** Firebase Test Lab XCTest archive (`qwen-iphone-probe-xctest.zip`) containing:
- `<scheme>.xctestrun` at the zip root
- `Release-iphoneos/QwenProbeHost.app` (embedding `llama.framework` from a pinned `ggml-org/llama.cpp` release)
- `Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest`

At test runtime (`QwenProbeTests`), twelve XCTest methods are available:
- `test1_clipCpu448`..`test6_q8AdapterTenLooks` keep the A13 options (`output.weight` / tied `token_embd.weight` on `ggml_backend_cpu_buffer_type()`, `cparams.op_offload = false`, every transformer layer on Metal):
  - `test1_clipCpu448`: vision encoder (CLIP) on the CPU with every online core, 448x448 image.
  - `test2_clipCpu768`: CLIP on the CPU with every online core, 576x768 image.
  - `test3_clipGpu768`: CLIP on Metal, 576x768 image.
  - `test4_clipCpu448Threads2`: CLIP on the CPU with 2 threads, 448x448 image.
  - `test5_appProfileTenLooks`: runs the Q4_K_M base + Q8_0 `mmproj` profile (CLIP on Metal, 576x768) ten times in a row.
  - `test6_q8AdapterTenLooks`: runs the official `Qwen3VL-2B-Instruct-Q8_0.gguf` base + Q8_0 `mmproj` + an F16 LoRA adapter (configured via `QWEN_PROBE_ADAPTER_URL` and `QWEN_PROBE_ADAPTER_BYTES`, applied at scale `1.0`) ten times in a row on a 576x768 image (`maxTokens = 64`).
- `test7_q8NoAdapterApp`..`test9_q8AdapterAppNoMmap` use Apple7+ GPU settings (every layer including `output.weight` on Metal, no CPU output override, `cparams.op_offload = true`, CLIP on Metal, 576x768, `maxTokens = 64`, 3 looks each):
  - `test7_q8NoAdapterApp`: `Qwen3VL-2B-Instruct-Q8_0.gguf` + Q8_0 `mmproj`, no LoRA adapter, default `mmap`.
  - `test8_q8AdapterApp`: `Qwen3VL-2B-Instruct-Q8_0.gguf` + Q8_0 `mmproj` + env-configured LoRA adapter at scale `1.0`, default `mmap`.
  - `test9_q8AdapterAppNoMmap`: as `test8_q8AdapterApp` with `use_mmap = false` (`LLAMA_LOAD_MODE_NONE`), logging post-load `phys_footprint` before the first look.
- `test10_q8AdapterAppUbatch128`..`test12_q8AdapterAppClipCpu` use the same Q8_0 base, Q8_0 `mmproj`, env-configured LoRA add-on, 576x768 image, `maxTokens = 64`, and 3 looks as `test8_q8AdapterApp`:
  - `test10_q8AdapterAppUbatch128`: Apple7+ app settings with `n_ubatch = 128` and default `mmap`.
  - `test11_q8AdapterAppUbatch128NoMmap`: as `test10_q8AdapterAppUbatch128` with `use_mmap = false`.
  - `test12_q8AdapterAppClipCpu`: Apple7+ app settings with default `n_ubatch` (`0`, llama.cpp default), default `mmap`, and CLIP on the CPU using `activeProcessorCount` threads.

Each look prints a single-line JSON report prefixed with `QWENPROBE_RESULT ` and attaches it as an `XCTAttachment` containing per-look timing (`encode_ms`, `first_token_ms`, `decode_tok_per_s`, `total_ms`, `tokens`), major page faults during decode (`decode_major_faults`, `decode_major_faults_per_token` via `getrusage(RUSAGE_SELF).ru_majflt`), thermal state (`thermal_state_before`, `thermal_state_after`), process memory (`os_proc_available_memory_before_load_bytes`, `lowest_os_proc_available_memory_bytes`, `phys_footprint_after_model_load_bytes`, `peak_phys_footprint_bytes`), and Metal memory (`metal_recommended_max_working_set_bytes`, `metal_current_allocated_after_load_bytes`, `metal_current_allocated_after_image_chunk_bytes`, `metal_current_allocated_after_decode_bytes`). The flushed `QWENPROBE_LOAD` line also prints `metal_recommended_max_working_set_bytes` and `metal_current_allocated_bytes` measured after load.

To select one or more test methods (executed in order) and inject runtime environment variables into a built `.zip` locally without rebuilding or printing values:

```bash
python3 scripts/set_probe_env.py dist/qwen-iphone-probe-xctest.zip dist/qwen-probe-q8b.zip \
  --only QwenProbeTests/test7_q8NoAdapterApp \
  --only QwenProbeTests/test8_q8AdapterApp \
  --only QwenProbeTests/test9_q8AdapterAppNoMmap \
  --only QwenProbeTests/test10_q8AdapterAppUbatch128 \
  --only QwenProbeTests/test11_q8AdapterAppUbatch128NoMmap \
  --only QwenProbeTests/test12_q8AdapterAppClipCpu \
  --env QWEN_PROBE_ADAPTER_URL=<adapter-gguf-url> \
  --env QWEN_PROBE_ADAPTER_BYTES=12860544
```

## License

MIT (see [LICENSE](LICENSE)). Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) at pinned release tag `b12020`, and downloads official [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (Apache-2.0) weights at test time.
