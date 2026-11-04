# qwen-iphone-probe

Minimal open-source (MIT) iOS XCTest probe to measure whether the official **Qwen3-VL-2B-Instruct** GGUF + vision projector (`mmproj`) loads and runs on a physical iPhone via **llama.cpp** (`mtmd` + Metal).

## What It Does

A GitHub Actions macOS workflow builds an ad-hoc signed **Release** Firebase Test Lab XCTest archive (`qwen-iphone-probe-xctest.zip`) containing:
- `<scheme>.xctestrun` at the zip root
- `Release-iphoneos/QwenProbeHost.app` (embedding `llama.framework` from a pinned `ggml-org/llama.cpp` release)
- `Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest`

At test runtime (`QwenProbeTests`), six XCTest methods are available:
- All methods keep the A13 crash fix: `output.weight` / tied `token_embd.weight` on `ggml_backend_cpu_buffer_type()` and `cparams.op_offload = false`, with every transformer layer on Metal.
- `test1_clipCpu448`: vision encoder (CLIP) on the CPU with every online core, 448x448 image.
- `test2_clipCpu768`: CLIP on the CPU with every online core, 576x768 image.
- `test3_clipGpu768`: CLIP on Metal, 576x768 image.
- `test4_clipCpu448Threads2`: CLIP on the CPU with 2 threads, 448x448 image.
- `test5_appProfileTenLooks`: runs the Q4_K_M base + Q8_0 `mmproj` profile (CLIP on Metal, 576x768) ten times in a row.
- `test6_q8AdapterTenLooks`: runs the official `Qwen3VL-2B-Instruct-Q8_0.gguf` base + Q8_0 `mmproj` + an F16 LoRA adapter (configured at test time via `QWEN_PROBE_ADAPTER_URL` and `QWEN_PROBE_ADAPTER_BYTES`, applied at scale `1.0`) ten times in a row on a 576x768 image with a multi-sentence description prompt (`maxTokens = 64`).

Each run prints a single-line JSON report prefixed with `QWENPROBE_RESULT ` and attaches it as an `XCTAttachment` containing per-look timing (`encode_ms`, `first_token_ms`, `decode_tok_per_s`, `total_ms`, `tokens`), thermal state (`thermal_state_before`, `thermal_state_after`), and memory metrics (`os_proc_available_memory_before_load_bytes`, `lowest_os_proc_available_memory_bytes`, `phys_footprint_after_model_load_bytes`, `peak_phys_footprint_bytes`).

To select a single test method and inject runtime environment variables into a built `.zip` locally without rebuilding or printing values:

```bash
python3 scripts/set_probe_env.py dist/qwen-iphone-probe-xctest.zip dist/qwen-probe-q8.zip \
  --only QwenProbeTests/test6_q8AdapterTenLooks \
  --env QWEN_PROBE_ADAPTER_URL=<adapter-gguf-url> \
  --env QWEN_PROBE_ADAPTER_BYTES=12860544
```

## License

MIT (see [LICENSE](LICENSE)). Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) at pinned release tag `b12020`, and downloads official [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (Apache-2.0) weights at test time.
