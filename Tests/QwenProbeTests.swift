import Darwin
import Foundation
import UIKit
import XCTest

private struct ModelSpec {
    let urlString: String
    let filename: String
    let expectedBytes: UInt64
}

private struct AdapterSpec {
    let urlString: String
    let cacheFilename: String
    let expectedBytes: UInt64
    let scale: Float
}

private let qwenProbeCLogger: @convention(c) (UnsafePointer<CChar>?) -> Void = { cStr in
    guard let cStr = cStr else { return }
    let msg = String(cString: cStr)
    print(msg)
    fflush(stdout)
    NSLog("%@", msg)
}

private struct QwenProbePayload: Encodable {
    let variant: String
    let look: Int?
    let device_model_identifier: String
    let ios_version: String
    let os_proc_available_memory_before_load_bytes: UInt64
    let lowest_os_proc_available_memory_bytes: UInt64
    let phys_footprint_after_model_load_bytes: UInt64
    let peak_phys_footprint_bytes: UInt64
    let decode_major_faults: Int64
    let decode_major_faults_per_token: Double
    let thermal_state_before: Int
    let thermal_state_after: Int
    let encode_ms: Double
    let first_token_ms: Double
    let decode_tok_per_s: Double
    let total_ms: Double
    let tokens: Int32
    let model_load_seconds: Double
    let image_encode_seconds: Double
    let image_encode_warm_seconds: Double
    let time_to_first_token_seconds: Double
    let total_seconds: Double
    let generated_tokens_per_second: Double
    let generated_tokens: Int32
    let generated_text: String
    let error: String?
    let model_url: String
    let model_bytes: UInt64
    let mmproj_url: String
    let mmproj_bytes: UInt64
    let adapter_bytes: UInt64
    let adapter_scale: Double

    enum CodingKeys: String, CodingKey {
        case variant
        case look
        case device_model_identifier
        case ios_version
        case os_proc_available_memory_before_load_bytes
        case lowest_os_proc_available_memory_bytes
        case phys_footprint_after_model_load_bytes
        case peak_phys_footprint_bytes
        case decode_major_faults
        case decode_major_faults_per_token
        case thermal_state_before
        case thermal_state_after
        case encode_ms
        case first_token_ms
        case decode_tok_per_s
        case total_ms
        case tokens
        case model_load_seconds
        case image_encode_seconds
        case image_encode_warm_seconds
        case time_to_first_token_seconds
        case total_seconds
        case generated_tokens_per_second
        case generated_tokens
        case generated_text
        case error
        case model_url
        case model_bytes
        case mmproj_url
        case mmproj_bytes
        case adapter_bytes
        case adapter_scale
    }

    private static func clean(_ value: Double) -> Double {
        return value.isFinite ? (value * 1_000_000.0).rounded() / 1_000_000.0 : 0.0
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(variant, forKey: .variant)
        try c.encodeIfPresent(look, forKey: .look)
        try c.encode(device_model_identifier, forKey: .device_model_identifier)
        try c.encode(ios_version, forKey: .ios_version)
        try c.encode(os_proc_available_memory_before_load_bytes, forKey: .os_proc_available_memory_before_load_bytes)
        try c.encode(lowest_os_proc_available_memory_bytes, forKey: .lowest_os_proc_available_memory_bytes)
        try c.encode(phys_footprint_after_model_load_bytes, forKey: .phys_footprint_after_model_load_bytes)
        try c.encode(peak_phys_footprint_bytes, forKey: .peak_phys_footprint_bytes)
        try c.encode(decode_major_faults, forKey: .decode_major_faults)
        try c.encode(Self.clean(decode_major_faults_per_token), forKey: .decode_major_faults_per_token)
        try c.encode(thermal_state_before, forKey: .thermal_state_before)
        try c.encode(thermal_state_after, forKey: .thermal_state_after)
        try c.encode(Self.clean(encode_ms), forKey: .encode_ms)
        try c.encode(Self.clean(first_token_ms), forKey: .first_token_ms)
        try c.encode(Self.clean(decode_tok_per_s), forKey: .decode_tok_per_s)
        try c.encode(Self.clean(total_ms), forKey: .total_ms)
        try c.encode(tokens, forKey: .tokens)
        try c.encode(Self.clean(model_load_seconds), forKey: .model_load_seconds)
        try c.encode(Self.clean(image_encode_seconds), forKey: .image_encode_seconds)
        try c.encode(Self.clean(image_encode_warm_seconds), forKey: .image_encode_warm_seconds)
        try c.encode(Self.clean(time_to_first_token_seconds), forKey: .time_to_first_token_seconds)
        try c.encode(Self.clean(total_seconds), forKey: .total_seconds)
        try c.encode(Self.clean(generated_tokens_per_second), forKey: .generated_tokens_per_second)
        try c.encode(generated_tokens, forKey: .generated_tokens)
        try c.encode(generated_text, forKey: .generated_text)
        try c.encode(error, forKey: .error)
        try c.encode(model_url, forKey: .model_url)
        try c.encode(model_bytes, forKey: .model_bytes)
        try c.encode(mmproj_url, forKey: .mmproj_url)
        try c.encode(mmproj_bytes, forKey: .mmproj_bytes)
        try c.encode(adapter_bytes, forKey: .adapter_bytes)
        try c.encode(Self.clean(adapter_scale), forKey: .adapter_scale)
    }
}

private final class ProbeSharedState {
    private let lock = NSLock()
    var payload: QwenProbePayload?
    var modelBytes: UInt64 = 0
    var mmprojBytes: UInt64 = 0
    var adapterBytes: UInt64 = 0
    var availBeforeLoad: UInt64 = 0

    func updateDownloadProgress(
        modelBytes: UInt64? = nil,
        mmprojBytes: UInt64? = nil,
        adapterBytes: UInt64? = nil,
        availBeforeLoad: UInt64? = nil
    ) {
        lock.lock()
        if let m = modelBytes { self.modelBytes = m }
        if let mm = mmprojBytes { self.mmprojBytes = mm }
        if let ad = adapterBytes { self.adapterBytes = ad }
        if let a = availBeforeLoad { self.availBeforeLoad = a }
        lock.unlock()
    }

    func setPayload(_ value: QwenProbePayload) {
        lock.lock()
        self.payload = value
        lock.unlock()
    }

    func snapshot() -> (
        payload: QwenProbePayload?,
        modelBytes: UInt64,
        mmprojBytes: UInt64,
        adapterBytes: UInt64,
        availBeforeLoad: UInt64
    ) {
        lock.lock()
        defer { lock.unlock() }
        return (payload, modelBytes, mmprojBytes, adapterBytes, availBeforeLoad)
    }
}

@objc(QwenProbeTests)
final class QwenProbeTests: XCTestCase {
    private static let modelURLString =
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    private static let modelFilename = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    private static let expectedModelBytes: UInt64 = 1_107_409_952

    private static let q8ModelURLString =
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let q8ModelFilename = "Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let expectedQ8ModelBytes: UInt64 = 1_834_427_424

    private static let mmprojURLString =
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let mmprojFilename = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let expectedMmprojBytes: UInt64 = 445_053_216

    private static let adapterURLEnvKey = "QWEN_PROBE_ADAPTER_URL"
    private static let adapterBytesEnvKey = "QWEN_PROBE_ADAPTER_BYTES"
    private static let adapterScale: Float = 1.0

    private static let promptText =
        "What colours and shapes are in this image? Answer in one sentence."
    private static let detailedPromptText =
        "Describe this image in detail, covering the foreground shape, its position, proportions, edges, and color contrast against the background in three sentences."
    private static let maxTokens: Int32 = 64
    private static let internalTimeoutSeconds: TimeInterval = 2400.0

    private static let defaultModelSpec = ModelSpec(
        urlString: modelURLString,
        filename: modelFilename,
        expectedBytes: expectedModelBytes
    )

    private static let q8ModelSpec = ModelSpec(
        urlString: q8ModelURLString,
        filename: q8ModelFilename,
        expectedBytes: expectedQ8ModelBytes
    )

    // All A13-profile methods keep the A13 crash fix (output head on CPU, op_offload off) and vary only
    // where the vision encoder runs and the image size. 576x768 is the app's longest side 768.
    private static func a13Options(clipCpu: Int32, clipThreads: Int32) -> qwen_probe_options {
        return qwen_probe_options(
            disable_flash_attn: 0,
            disable_mmap: 0,
            disable_op_offload: 1,
            cpu_output_override: 1,
            clip_cpu: clipCpu,
            clip_threads: clipThreads
        )
    }

    // App settings for Apple7+ GPUs: every layer on Metal (n_gpu_layers = -1),
    // no CPU output override (cpu_output_override = 0), and op_offload on (disable_op_offload = 0).
    private static func apple7AppOptions(useMmap: Bool = true) -> qwen_probe_options {
        return qwen_probe_options(
            disable_flash_attn: 0,
            disable_mmap: useMmap ? 0 : 1,
            disable_op_offload: 0,
            cpu_output_override: 0,
            clip_cpu: 0,
            clip_threads: 0
        )
    }

    private func requireAdapterSpecFromEnv(testName: String, variantTag: String) -> AdapterSpec? {
        continueAfterFailure = false
        executionTimeAllowance = 2400

        let env = ProcessInfo.processInfo.environment
        let rawAdapterURL = (env[Self.adapterURLEnvKey] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let rawAdapterBytes = (env[Self.adapterBytesEnvKey] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

        var missingOrInvalid: [String] = []
        if rawAdapterURL.isEmpty {
            missingOrInvalid.append("\(Self.adapterURLEnvKey) is missing or empty")
        } else if let parsed = URL(string: rawAdapterURL),
                  let scheme = parsed.scheme?.lowercased(),
                  (scheme == "https" || scheme == "http"),
                  parsed.host != nil {
            // Valid HTTP(S) URL structure.
        } else {
            missingOrInvalid.append("\(Self.adapterURLEnvKey) is not a valid HTTP(S) URL")
        }

        var expectedAdapterBytes: UInt64 = 0
        if rawAdapterBytes.isEmpty {
            missingOrInvalid.append("\(Self.adapterBytesEnvKey) is missing or empty")
        } else if let parsedBytes = UInt64(rawAdapterBytes), parsedBytes > 0 {
            expectedAdapterBytes = parsedBytes
        } else {
            missingOrInvalid.append("\(Self.adapterBytesEnvKey) must be a positive integer byte count")
        }

        if !missingOrInvalid.isEmpty {
            let message = "Refusing to run \(testName) without LoRA add-on: " +
                missingOrInvalid.joined(separator: "; ")
            Self.logLine("[QWENPROBE] variant=\(variantTag) step=env_check_failed error=\(message)")
            XCTFail(message)
            return nil
        }

        return AdapterSpec(
            urlString: rawAdapterURL,
            cacheFilename: Self.adapterCacheFilename(urlString: rawAdapterURL, expectedBytes: expectedAdapterBytes),
            expectedBytes: expectedAdapterBytes,
            scale: Self.adapterScale
        )
    }

    @objc func test1_clipCpu448() {
        runProbeVariant(variant: "clip_cpu_448_all_cores",
                        options: Self.a13Options(clipCpu: 1, clipThreads: -1), width: 448, height: 448)
    }

    @objc func test2_clipCpu768() {
        runProbeVariant(variant: "clip_cpu_576x768_all_cores",
                        options: Self.a13Options(clipCpu: 1, clipThreads: -1), width: 576, height: 768)
    }

    @objc func test3_clipGpu768() {
        runProbeVariant(variant: "clip_gpu_576x768",
                        options: Self.a13Options(clipCpu: 0, clipThreads: 0), width: 576, height: 768)
    }

    @objc func test4_clipCpu448Threads2() {
        runProbeVariant(variant: "clip_cpu_448_2_threads",
                        options: Self.a13Options(clipCpu: 1, clipThreads: 2), width: 448, height: 448)
    }

    // Ten back-to-back looks with the app profile, to see slowdown and heat on one device.
    // thermalState raw values: 0 nominal, 1 fair, 2 serious, 3 critical.
    @objc func test5_appProfileTenLooks() {
        runProbeVariant(
            variant: "app_profile",
            options: Self.a13Options(clipCpu: 0, clipThreads: 0),
            width: 576,
            height: 768,
            looks: 10
        )
    }

    // Ten back-to-back looks with the Q8_0 base, Q8_0 mmproj, and F16 LoRA add-on at scale 1.0 (A13 options).
    @objc func test6_q8AdapterTenLooks() {
        guard let adapterSpec = requireAdapterSpecFromEnv(
            testName: "test6_q8AdapterTenLooks",
            variantTag: "q8_adapter"
        ) else {
            return
        }
        runProbeVariant(
            variant: "q8_adapter",
            options: Self.a13Options(clipCpu: 0, clipThreads: 0),
            width: 576,
            height: 768,
            looks: 10,
            modelSpec: Self.q8ModelSpec,
            adapterSpec: adapterSpec,
            prompt: Self.detailedPromptText,
            maxTokens: Self.maxTokens
        )
    }

    // Three looks: Q8_0 base + Q8_0 mmproj, no add-on, Apple7+ GPU app settings, default mmap.
    @objc func test7_q8NoAdapterApp() {
        runProbeVariant(
            variant: "q8_no_adapter_app",
            options: Self.apple7AppOptions(useMmap: true),
            width: 576,
            height: 768,
            looks: 3,
            modelSpec: Self.q8ModelSpec,
            adapterSpec: nil,
            prompt: Self.detailedPromptText,
            maxTokens: Self.maxTokens
        )
    }

    // Three looks: Q8_0 base + Q8_0 mmproj + env LoRA add-on, Apple7+ GPU app settings, default mmap.
    @objc func test8_q8AdapterApp() {
        guard let adapterSpec = requireAdapterSpecFromEnv(
            testName: "test8_q8AdapterApp",
            variantTag: "q8_adapter_app"
        ) else {
            return
        }
        runProbeVariant(
            variant: "q8_adapter_app",
            options: Self.apple7AppOptions(useMmap: true),
            width: 576,
            height: 768,
            looks: 3,
            modelSpec: Self.q8ModelSpec,
            adapterSpec: adapterSpec,
            prompt: Self.detailedPromptText,
            maxTokens: Self.maxTokens
        )
    }

    // Three looks: as test8 with use_mmap = false (LLAMA_LOAD_MODE_NONE); logs post-load footprint before look 1.
    @objc func test9_q8AdapterAppNoMmap() {
        guard let adapterSpec = requireAdapterSpecFromEnv(
            testName: "test9_q8AdapterAppNoMmap",
            variantTag: "q8_adapter_app_no_mmap"
        ) else {
            return
        }
        runProbeVariant(
            variant: "q8_adapter_app_no_mmap",
            options: Self.apple7AppOptions(useMmap: false),
            width: 576,
            height: 768,
            looks: 3,
            modelSpec: Self.q8ModelSpec,
            adapterSpec: adapterSpec,
            prompt: Self.detailedPromptText,
            maxTokens: Self.maxTokens
        )
    }

    private func runProbeVariant(
        variant: String,
        options: qwen_probe_options,
        width: Int,
        height: Int,
        looks: Int = 1,
        modelSpec: ModelSpec = QwenProbeTests.defaultModelSpec,
        adapterSpec: AdapterSpec? = nil,
        prompt: String = QwenProbeTests.promptText,
        maxTokens: Int32 = QwenProbeTests.maxTokens
    ) {
        continueAfterFailure = false
        executionTimeAllowance = 2400

        if looks <= 1 {
            runSingleLook(
                variant: variant,
                options: options,
                width: width,
                height: height,
                lookIndex: nil,
                thermalBeforeOverride: nil,
                modelSpec: modelSpec,
                adapterSpec: adapterSpec,
                prompt: prompt,
                maxTokens: maxTokens
            )
            return
        }

        for look in 1...looks {
            let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
            let start = Date()
            let lookVariant = String(format: "%@_look_%02d", variant, look)
            runSingleLook(
                variant: lookVariant,
                options: options,
                width: width,
                height: height,
                lookIndex: look,
                thermalBeforeOverride: thermalBefore,
                modelSpec: modelSpec,
                adapterSpec: adapterSpec,
                prompt: prompt,
                maxTokens: maxTokens
            )
            Self.logLine(
                "QWENPROBE_LOOK look=\(look) wall_seconds=\(String(format: "%.3f", Date().timeIntervalSince(start))) " +
                "thermal_before=\(thermalBefore) thermal_after=\(ProcessInfo.processInfo.thermalState.rawValue) " +
                "variant=\(variant) looks=\(looks)"
            )
        }
    }

    private func runSingleLook(
        variant: String,
        options: qwen_probe_options,
        width: Int,
        height: Int,
        lookIndex: Int? = nil,
        thermalBeforeOverride: Int? = nil,
        modelSpec: ModelSpec = QwenProbeTests.defaultModelSpec,
        adapterSpec: AdapterSpec? = nil,
        prompt: String = QwenProbeTests.promptText,
        maxTokens: Int32 = QwenProbeTests.maxTokens
    ) {
        continueAfterFailure = false
        executionTimeAllowance = 2400
        qwen_probe_set_logger(qwenProbeCLogger)

        let iosVersion: String
        if Thread.isMainThread {
            UIApplication.shared.isIdleTimerDisabled = true
            iosVersion = UIDevice.current.systemVersion
        } else {
            var ver = "unknown"
            DispatchQueue.main.sync {
                UIApplication.shared.isIdleTimerDisabled = true
                ver = UIDevice.current.systemVersion
            }
            iosVersion = ver
        }

        let thermalBefore = thermalBeforeOverride ?? ProcessInfo.processInfo.thermalState.rawValue
        let deviceModel = Self.readDeviceModelIdentifier()
        qwen_probe_reset_live_memory_stats()
        let initialAvail = qwen_probe_available_memory()

        Self.logLine(
            "[QWENPROBE] variant=\(variant) step=test_start device=\(deviceModel) ios=\(iosVersion) " +
            "thermal_before=\(thermalBefore) avail_bytes=\(initialAvail) " +
            "phys_footprint_bytes=\(qwen_probe_phys_footprint())"
        )

        let state = ProbeSharedState()
        state.updateDownloadProgress(availBeforeLoad: initialAvail)
        let activeAdapterScale = Double(adapterSpec?.scale ?? 0.0)

        let doneExp = expectation(description: "Qwen3-VL-2B probe (\(variant))")

        DispatchQueue.global(qos: .userInitiated).async {
            autoreleasepool {
                defer {
                    doneExp.fulfill()
                }

                let cacheDir: URL
                do {
                    cacheDir = try Self.modelsCacheDirectory()
                } catch {
                    let availNow = qwen_probe_available_memory()
                    let lowestNow = qwen_probe_live_lowest_available_memory()
                    let failPayload = QwenProbePayload(
                        variant: variant,
                        look: lookIndex,
                        device_model_identifier: deviceModel,
                        ios_version: iosVersion,
                        os_proc_available_memory_before_load_bytes: availNow,
                        lowest_os_proc_available_memory_bytes: lowestNow > 0 ? lowestNow : availNow,
                        phys_footprint_after_model_load_bytes: 0,
                        peak_phys_footprint_bytes: qwen_probe_live_peak_footprint(),
                        decode_major_faults: 0,
                        decode_major_faults_per_token: 0.0,
                        thermal_state_before: thermalBefore,
                        thermal_state_after: ProcessInfo.processInfo.thermalState.rawValue,
                        encode_ms: 0.0,
                        first_token_ms: 0.0,
                        decode_tok_per_s: 0.0,
                        total_ms: 0.0,
                        tokens: 0,
                        model_load_seconds: 0.0,
                        image_encode_seconds: 0.0,
                        image_encode_warm_seconds: 0.0,
                        time_to_first_token_seconds: 0.0,
                        total_seconds: 0.0,
                        generated_tokens_per_second: 0.0,
                        generated_tokens: 0,
                        generated_text: "",
                        error: "Failed to prepare Caches directory: \(error.localizedDescription)",
                        model_url: modelSpec.urlString,
                        model_bytes: 0,
                        mmproj_url: Self.mmprojURLString,
                        mmproj_bytes: 0,
                        adapter_bytes: 0,
                        adapter_scale: activeAdapterScale
                    )
                    state.setPayload(failPayload)
                    return
                }

                let modelDest = cacheDir.appendingPathComponent(modelSpec.filename)
                let mmprojDest = cacheDir.appendingPathComponent(Self.mmprojFilename)
                let adapterDest = adapterSpec.map { cacheDir.appendingPathComponent($0.cacheFilename) }

                var cachedModelBytes: UInt64 = 0
                var cachedMmprojBytes: UInt64 = 0
                var cachedAdapterBytes: UInt64 = 0

                do {
                    cachedModelBytes = try Self.ensureCachedGGUF(
                        variant: variant,
                        label: "model",
                        urlString: modelSpec.urlString,
                        expectedBytes: modelSpec.expectedBytes,
                        destination: modelDest,
                        redactURL: false
                    )
                    state.updateDownloadProgress(modelBytes: cachedModelBytes)
                    qwen_probe_record_live_footprint()

                    cachedMmprojBytes = try Self.ensureCachedGGUF(
                        variant: variant,
                        label: "mmproj",
                        urlString: Self.mmprojURLString,
                        expectedBytes: Self.expectedMmprojBytes,
                        destination: mmprojDest,
                        redactURL: false
                    )
                    state.updateDownloadProgress(mmprojBytes: cachedMmprojBytes)
                    qwen_probe_record_live_footprint()

                    if let spec = adapterSpec, let dest = adapterDest {
                        cachedAdapterBytes = try Self.ensureCachedGGUF(
                            variant: variant,
                            label: "adapter",
                            urlString: spec.urlString,
                            expectedBytes: spec.expectedBytes,
                            destination: dest,
                            redactURL: true
                        )
                        state.updateDownloadProgress(adapterBytes: cachedAdapterBytes)
                        qwen_probe_record_live_footprint()
                    }
                } catch {
                    let availNow = qwen_probe_available_memory()
                    let lowestNow = qwen_probe_live_lowest_available_memory()
                    let peakNow = qwen_probe_live_peak_footprint()
                    let scrubbedErr: String
                    if let spec = adapterSpec {
                        scrubbedErr = error.localizedDescription.replacingOccurrences(of: spec.urlString, with: "<redacted>")
                    } else {
                        scrubbedErr = error.localizedDescription
                    }
                    let failPayload = QwenProbePayload(
                        variant: variant,
                        look: lookIndex,
                        device_model_identifier: deviceModel,
                        ios_version: iosVersion,
                        os_proc_available_memory_before_load_bytes: availNow,
                        lowest_os_proc_available_memory_bytes: lowestNow > 0 ? lowestNow : availNow,
                        phys_footprint_after_model_load_bytes: 0,
                        peak_phys_footprint_bytes: peakNow,
                        decode_major_faults: 0,
                        decode_major_faults_per_token: 0.0,
                        thermal_state_before: thermalBefore,
                        thermal_state_after: ProcessInfo.processInfo.thermalState.rawValue,
                        encode_ms: 0.0,
                        first_token_ms: 0.0,
                        decode_tok_per_s: 0.0,
                        total_ms: 0.0,
                        tokens: 0,
                        model_load_seconds: 0.0,
                        image_encode_seconds: 0.0,
                        image_encode_warm_seconds: 0.0,
                        time_to_first_token_seconds: 0.0,
                        total_seconds: 0.0,
                        generated_tokens_per_second: 0.0,
                        generated_tokens: 0,
                        generated_text: "",
                        error: "Download failed: \(scrubbedErr)",
                        model_url: modelSpec.urlString,
                        model_bytes: cachedModelBytes,
                        mmproj_url: Self.mmprojURLString,
                        mmproj_bytes: cachedMmprojBytes,
                        adapter_bytes: cachedAdapterBytes,
                        adapter_scale: activeAdapterScale
                    )
                    state.setPayload(failPayload)
                    return
                }

                let rgbBytes = Self.makeSyntheticRGBImage(width: width, height: height)
                qwen_probe_reset_live_memory_stats()
                let availBefore = qwen_probe_available_memory()
                state.updateDownloadProgress(availBeforeLoad: availBefore)

                let metricsPtr = UnsafeMutablePointer<qwen_probe_metrics>.allocate(capacity: 1)
                metricsPtr.initialize(to: qwen_probe_metrics())
                defer {
                    metricsPtr.deinitialize(count: 1)
                    metricsPtr.deallocate()
                }

                var textBuf = [CChar](repeating: 0, count: 8192)
                var errBuf = [CChar](repeating: 0, count: 2048)

                let modelPath = modelDest.path
                let mmprojPath = mmprojDest.path
                let adapterPath = adapterDest?.path ?? ""
                let loraScale = adapterSpec?.scale ?? 0.0
                var optsCopy = options

                Self.logLine(
                    "[QWENPROBE] variant=\(variant) step=starting_engine width=\(width) " +
                    "height=\(height) max_tokens=\(maxTokens) has_adapter=\(adapterSpec != nil ? 1 : 0) " +
                    "phys_footprint_bytes=\(qwen_probe_phys_footprint())"
                )

                let rc = Self.invokeEngine(
                    variant: variant,
                    options: &optsCopy,
                    modelPath: modelPath,
                    mmprojPath: mmprojPath,
                    adapterPath: adapterPath,
                    adapterScale: loraScale,
                    rgbBytes: rgbBytes,
                    width: width,
                    height: height,
                    prompt: prompt,
                    maxTokens: maxTokens,
                    metricsPtr: metricsPtr,
                    textBuf: &textBuf,
                    errBuf: &errBuf
                )

                let thermalAfter = ProcessInfo.processInfo.thermalState.rawValue
                let metrics = metricsPtr.pointee
                let rawGenerated = textBuf.withUnsafeBufferPointer { ptr in
                    String(cString: ptr.baseAddress!)
                }
                let generatedText = rawGenerated.trimmingCharacters(in: .whitespacesAndNewlines)
                let engineError = errBuf.withUnsafeBufferPointer { ptr in
                    String(cString: ptr.baseAddress!)
                }.trimmingCharacters(in: .whitespacesAndNewlines)

                let errorMessage: String?
                if rc != 0 {
                    errorMessage = engineError.isEmpty ? "qwen_probe_run_with_adapter failed with code \(rc)" : engineError
                } else if generatedText.isEmpty {
                    errorMessage = "Model loaded and ran without error code, but generated_text was empty"
                } else {
                    errorMessage = nil
                }

                let peakFootprint = max(metrics.peak_phys_footprint_bytes, qwen_probe_live_peak_footprint())
                let availRecorded = metrics.avail_mem_before_load_bytes > 0
                    ? metrics.avail_mem_before_load_bytes
                    : availBefore
                let liveLowest = qwen_probe_live_lowest_available_memory()
                let lowestCandidates: [UInt64] = [metrics.lowest_avail_mem_bytes, liveLowest, availRecorded].filter { $0 > 0 }
                let lowestAvail: UInt64 = lowestCandidates.min() ?? 0
                let afterLoadFootprint: UInt64 = metrics.phys_footprint_after_model_load_bytes
                let decMajFlt: Int64 = metrics.decode_major_faults
                let decMajFltPerTok: Double = metrics.decode_major_faults_per_token

                let finalPayload = QwenProbePayload(
                    variant: variant,
                    look: lookIndex,
                    device_model_identifier: deviceModel,
                    ios_version: iosVersion,
                    os_proc_available_memory_before_load_bytes: availRecorded,
                    lowest_os_proc_available_memory_bytes: lowestAvail,
                    phys_footprint_after_model_load_bytes: afterLoadFootprint,
                    peak_phys_footprint_bytes: peakFootprint,
                    decode_major_faults: decMajFlt,
                    decode_major_faults_per_token: decMajFltPerTok,
                    thermal_state_before: thermalBefore,
                    thermal_state_after: thermalAfter,
                    encode_ms: metrics.image_encode_seconds * 1000.0,
                    first_token_ms: metrics.time_to_first_token_seconds * 1000.0,
                    decode_tok_per_s: metrics.generated_tokens_per_second,
                    total_ms: metrics.total_inference_seconds * 1000.0,
                    tokens: metrics.generated_tokens,
                    model_load_seconds: metrics.model_load_seconds,
                    image_encode_seconds: metrics.image_encode_seconds,
                    image_encode_warm_seconds: metrics.image_encode_warm_seconds,
                    time_to_first_token_seconds: metrics.time_to_first_token_seconds,
                    total_seconds: metrics.total_inference_seconds,
                    generated_tokens_per_second: metrics.generated_tokens_per_second,
                    generated_tokens: metrics.generated_tokens,
                    generated_text: generatedText,
                    error: errorMessage,
                    model_url: modelSpec.urlString,
                    model_bytes: cachedModelBytes,
                    mmproj_url: Self.mmprojURLString,
                    mmproj_bytes: cachedMmprojBytes,
                    adapter_bytes: cachedAdapterBytes,
                    adapter_scale: activeAdapterScale
                )
                state.setPayload(finalPayload)
            }
        }

        let waitResult = XCTWaiter.wait(for: [doneExp], timeout: Self.internalTimeoutSeconds)
        let snap = state.snapshot()

        let finalPayload: QwenProbePayload
        if let completedPayload = snap.payload {
            finalPayload = completedPayload
        } else {
            let timeoutReason = (waitResult == .timedOut)
                ? "Timed out after \(Int(Self.internalTimeoutSeconds))s waiting for \(variant)"
                : "XCTWaiter ended with status \(waitResult.rawValue) before payload was produced for \(variant)"
            let availNow = qwen_probe_available_memory()
            let availStart = snap.availBeforeLoad > 0 ? snap.availBeforeLoad : availNow
            let liveLowest = qwen_probe_live_lowest_available_memory()
            let lowestCandidates: [UInt64] = [liveLowest, availNow, availStart].filter { $0 > 0 }
            let lowestAvail: UInt64 = lowestCandidates.min() ?? 0
            let liveAfterLoadFp: UInt64 = qwen_probe_live_after_load_footprint()
            finalPayload = QwenProbePayload(
                variant: variant,
                look: lookIndex,
                device_model_identifier: deviceModel,
                ios_version: iosVersion,
                os_proc_available_memory_before_load_bytes: availStart,
                lowest_os_proc_available_memory_bytes: lowestAvail,
                phys_footprint_after_model_load_bytes: liveAfterLoadFp,
                peak_phys_footprint_bytes: qwen_probe_live_peak_footprint(),
                decode_major_faults: 0,
                decode_major_faults_per_token: 0.0,
                thermal_state_before: thermalBefore,
                thermal_state_after: ProcessInfo.processInfo.thermalState.rawValue,
                encode_ms: 0.0,
                first_token_ms: 0.0,
                decode_tok_per_s: 0.0,
                total_ms: 0.0,
                tokens: 0,
                model_load_seconds: 0.0,
                image_encode_seconds: 0.0,
                image_encode_warm_seconds: 0.0,
                time_to_first_token_seconds: 0.0,
                total_seconds: 0.0,
                generated_tokens_per_second: 0.0,
                generated_tokens: 0,
                generated_text: "",
                error: timeoutReason,
                model_url: modelSpec.urlString,
                model_bytes: snap.modelBytes,
                mmproj_url: Self.mmprojURLString,
                mmproj_bytes: snap.mmprojBytes,
                adapter_bytes: snap.adapterBytes,
                adapter_scale: activeAdapterScale
            )
        }

        let jsonString = Self.encodeSingleLineJSON(finalPayload)
        let resultLine = "QWENPROBE_RESULT \(jsonString)"

        print(resultLine)
        fflush(stdout)
        NSLog("%@", resultLine)

        let lineAttachment = XCTAttachment(string: resultLine)
        lineAttachment.name = "QWENPROBE_RESULT_\(variant).txt"
        lineAttachment.lifetime = .keepAlways
        self.add(lineAttachment)

        let jsonAttachment = XCTAttachment(
            data: Data(jsonString.utf8),
            uniformTypeIdentifier: "public.json"
        )
        jsonAttachment.name = "qwenprobe_result_\(variant).json"
        jsonAttachment.lifetime = .keepAlways
        self.add(jsonAttachment)

        if let err = finalPayload.error {
            XCTFail("QWENPROBE (\(variant)) failed: \(err)")
        }
        XCTAssertFalse(finalPayload.generated_text.isEmpty, "Generated text must be non-empty (\(variant))")
    }

    private static func invokeEngine(
        variant: String,
        options: inout qwen_probe_options,
        modelPath: String,
        mmprojPath: String,
        adapterPath: String,
        adapterScale: Float,
        rgbBytes: [UInt8],
        width: Int,
        height: Int,
        prompt: String,
        maxTokens: Int32,
        metricsPtr: UnsafeMutablePointer<qwen_probe_metrics>,
        textBuf: inout [CChar],
        errBuf: inout [CChar]
    ) -> Int32 {
        let textCap = textBuf.count
        let errCap = errBuf.count
        return withUnsafePointer(to: &options) { optsPtr -> Int32 in
            rgbBytes.withUnsafeBufferPointer { rgbPtr -> Int32 in
                textBuf.withUnsafeMutableBufferPointer { textPtr -> Int32 in
                    errBuf.withUnsafeMutableBufferPointer { errPtr -> Int32 in
                        variant.withCString { variantCStr -> Int32 in
                            modelPath.withCString { modelCStr -> Int32 in
                                mmprojPath.withCString { mmprojCStr -> Int32 in
                                    adapterPath.withCString { adapterCStr -> Int32 in
                                        prompt.withCString { promptCStr -> Int32 in
                                            qwen_probe_run_with_adapter(
                                                variantCStr,
                                                optsPtr,
                                                modelCStr,
                                                mmprojCStr,
                                                adapterCStr,
                                                adapterScale,
                                                rgbPtr.baseAddress,
                                                UInt32(width),
                                                UInt32(height),
                                                promptCStr,
                                                maxTokens,
                                                metricsPtr,
                                                textPtr.baseAddress,
                                                textCap,
                                                errPtr.baseAddress,
                                                errCap
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static func adapterCacheFilename(urlString: String, expectedBytes: UInt64) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in urlString.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        let hex = String(hash, radix: 16)
        let padded = String(repeating: "0", count: max(0, 16 - hex.count)) + hex
        return "qwenprobe-adapter-\(padded)-\(expectedBytes).gguf"
    }

    private static func modelsCacheDirectory() throws -> URL {
        guard let cachesRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw NSError(
                domain: "QwenProbeCache",
                code: -10,
                userInfo: [NSLocalizedDescriptionKey: "Unable to locate Library/Caches directory"]
            )
        }
        let dir = cachesRoot.appendingPathComponent("qwenprobe-models", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableDir = dir
        try? mutableDir.setResourceValues(values)
        return dir
    }

    private static func validateCachedGGUF(at url: URL, expectedBytes: UInt64) -> UInt64? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let sizeNum = attrs[.size] as? NSNumber else {
            return nil
        }
        let fileSize = sizeNum.uint64Value
        guard fileSize == expectedBytes else {
            return nil
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        let magicData = (try? handle.read(upToCount: 4)) ?? Data()
        try? handle.close()
        let ggufMagic = Data([0x47, 0x47, 0x55, 0x46]) // "GGUF"
        guard magicData == ggufMagic else {
            return nil
        }
        return fileSize
    }

    private static func ensureCachedGGUF(
        variant: String,
        label: String,
        urlString: String,
        expectedBytes: UInt64,
        destination: URL,
        redactURL: Bool
    ) throws -> UInt64 {
        if let cachedBytes = validateCachedGGUF(at: destination, expectedBytes: expectedBytes) {
            logLine(
                "[QWENPROBE] variant=\(variant) step=cache_hit label=\(label) " +
                "path=\(destination.path) bytes=\(cachedBytes) " +
                "phys_footprint_bytes=\(qwen_probe_phys_footprint())"
            )
            return cachedBytes
        }

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.urlCache = nil
        sessionConfig.requestCachePolicy = .reloadIgnoringLocalCacheData
        sessionConfig.timeoutIntervalForRequest = 300
        sessionConfig.timeoutIntervalForResource = 1800
        sessionConfig.waitsForConnectivity = true
        let session = URLSession(configuration: sessionConfig)
        defer {
            session.finishTasksAndInvalidate()
        }

        return try downloadGGUF(
            session: session,
            variant: variant,
            label: label,
            urlString: urlString,
            expectedBytes: expectedBytes,
            destination: destination,
            redactURL: redactURL
        )
    }

    private static func readDeviceModelIdentifier() -> String {
        var machineBuf = [CChar](repeating: 0, count: 128)
        machineBuf.withUnsafeMutableBufferPointer { ptr in
            qwen_probe_device_machine(ptr.baseAddress, ptr.count)
        }
        return machineBuf.withUnsafeBufferPointer { ptr in
            String(cString: ptr.baseAddress!)
        }
    }

    private static func encodeSingleLineJSON(_ payload: QwenProbePayload) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(payload),
           let str = String(data: data, encoding: .utf8) {
            return str
                .replacingOccurrences(of: "\r", with: "")
                .replacingOccurrences(of: "\n", with: "")
        }
        let escapedError = (payload.error ?? "JSON encoding failed")
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: " ")
        let fallbackFields: [String] = [
            "\"variant\":\"\(payload.variant)\"",
            "\"device_model_identifier\":\"\(payload.device_model_identifier)\"",
            "\"ios_version\":\"\(payload.ios_version)\"",
            "\"os_proc_available_memory_before_load_bytes\":\(payload.os_proc_available_memory_before_load_bytes)",
            "\"lowest_os_proc_available_memory_bytes\":\(payload.lowest_os_proc_available_memory_bytes)",
            "\"phys_footprint_after_model_load_bytes\":\(payload.phys_footprint_after_model_load_bytes)",
            "\"peak_phys_footprint_bytes\":\(payload.peak_phys_footprint_bytes)",
            "\"decode_major_faults\":\(payload.decode_major_faults)",
            "\"decode_major_faults_per_token\":\(Self.clean(payload.decode_major_faults_per_token))",
            "\"thermal_state_before\":\(payload.thermal_state_before)",
            "\"thermal_state_after\":\(payload.thermal_state_after)",
            "\"generated_text\":\"\"",
            "\"error\":\"\(escapedError)\""
        ]
        return "{" + fallbackFields.joined(separator: ",") + "}"
    }

    private static func logLine(_ message: String) {
        print(message)
        fflush(stdout)
        NSLog("%@", message)
    }

    private static func downloadGGUF(
        session: URLSession,
        variant: String,
        label: String,
        urlString: String,
        expectedBytes: UInt64,
        destination: URL,
        redactURL: Bool
    ) throws -> UInt64 {
        let displayURL = redactURL ? "<redacted>" : urlString
        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "QwenProbeDownload",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid \(label) URL: \(displayURL)"]
            )
        }

        let partURL = destination.appendingPathExtension("part")
        let maxAttempts = 3
        var lastError: Error?

        for attempt in 1...maxAttempts {
            try? FileManager.default.removeItem(at: partURL)
            logLine(
                "[QWENPROBE] variant=\(variant) step=download_start label=\(label) " +
                "attempt=\(attempt)/\(maxAttempts) url=\(displayURL) expected_bytes=\(expectedBytes) " +
                "phys_footprint_bytes=\(qwen_probe_phys_footprint())"
            )

            let t0 = Date()
            let sema = DispatchSemaphore(value: 0)
            var taskError: Error?
            var httpStatus: Int = 0
            var expectedContentLen: Int64 = -1

            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.cachePolicy = .reloadIgnoringLocalCacheData

            let task = session.downloadTask(with: request) { tempURL, response, error in
                defer { sema.signal() }
                if let error = error {
                    taskError = error
                    return
                }
                if let http = response as? HTTPURLResponse {
                    httpStatus = http.statusCode
                    expectedContentLen = http.expectedContentLength
                    if !(200...299).contains(http.statusCode) {
                        taskError = NSError(
                            domain: "QwenProbeDownload",
                            code: http.statusCode,
                            userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) for \(displayURL)"]
                        )
                        return
                    }
                }
                guard let tempURL = tempURL else {
                    taskError = NSError(
                        domain: "QwenProbeDownload",
                        code: -2,
                        userInfo: [NSLocalizedDescriptionKey: "Missing temporary download file for \(label)"]
                    )
                    return
                }
                do {
                    try? FileManager.default.removeItem(at: partURL)
                    try FileManager.default.moveItem(at: tempURL, to: partURL)
                } catch {
                    taskError = error
                }
            }
            task.resume()
            sema.wait()

            if let err = taskError {
                lastError = err
                try? FileManager.default.removeItem(at: partURL)
                let scrubbed = redactURL
                    ? err.localizedDescription.replacingOccurrences(of: urlString, with: "<redacted>")
                    : err.localizedDescription
                logLine("[QWENPROBE] variant=\(variant) Download \(label) attempt=\(attempt) failed: \(scrubbed)")
                if attempt < maxAttempts {
                    Thread.sleep(forTimeInterval: TimeInterval(attempt * 3))
                }
                continue
            }

            do {
                let attrs = try FileManager.default.attributesOfItem(atPath: partURL.path)
                let fileSize = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
                if fileSize < 1024 {
                    throw NSError(
                        domain: "QwenProbeDownload",
                        code: -3,
                        userInfo: [NSLocalizedDescriptionKey: "Downloaded \(label) too small (\(fileSize) bytes)"]
                    )
                }
                if expectedContentLen > 0 && UInt64(expectedContentLen) != fileSize {
                    throw NSError(
                        domain: "QwenProbeDownload",
                        code: -4,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Incomplete \(label) download: got \(fileSize) bytes, Content-Length was \(expectedContentLen)"
                        ]
                    )
                }
                if fileSize != expectedBytes {
                    throw NSError(
                        domain: "QwenProbeDownload",
                        code: -6,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Size mismatch for \(label): got \(fileSize) bytes, expected \(expectedBytes)"
                        ]
                    )
                }

                let handle = try FileHandle(forReadingFrom: partURL)
                let magicData = try handle.read(upToCount: 4) ?? Data()
                try? handle.close()
                let ggufMagic = Data([0x47, 0x47, 0x55, 0x46]) // "GGUF"
                if magicData != ggufMagic {
                    throw NSError(
                        domain: "QwenProbeDownload",
                        code: -5,
                        userInfo: [NSLocalizedDescriptionKey: "Downloaded \(label) does not start with GGUF magic"]
                    )
                }

                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: partURL, to: destination)

                var values = URLResourceValues()
                values.isExcludedFromBackup = true
                var mutableDest = destination
                try? mutableDest.setResourceValues(values)

                let elapsed = Date().timeIntervalSince(t0)
                logLine(
                    "[QWENPROBE] variant=\(variant) step=download_done label=\(label) http=\(httpStatus) " +
                    "path=\(destination.path) bytes=\(fileSize) expected_bytes=\(expectedBytes) " +
                    "elapsed_s=\(String(format: "%.2f", elapsed)) phys_footprint_bytes=\(qwen_probe_phys_footprint())"
                )
                return fileSize
            } catch {
                lastError = error
                try? FileManager.default.removeItem(at: partURL)
                let scrubbed = redactURL
                    ? error.localizedDescription.replacingOccurrences(of: urlString, with: "<redacted>")
                    : error.localizedDescription
                logLine("[QWENPROBE] variant=\(variant) Validation \(label) attempt=\(attempt) failed: \(scrubbed)")
                if attempt < maxAttempts {
                    Thread.sleep(forTimeInterval: TimeInterval(attempt * 3))
                }
            }
        }

        throw lastError ?? NSError(
            domain: "QwenProbeDownload",
            code: -99,
            userInfo: [NSLocalizedDescriptionKey: "Exhausted retries downloading \(label) from \(displayURL)"]
        )
    }

    private static func makeSyntheticRGBImage(width: Int, height: Int) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let sqMinX = width / 4
        let sqMaxX = (width * 3) / 4
        let sqMinY = height / 4
        let sqMaxY = (height * 3) / 4

        for y in 0..<height {
            let inSquareY = (y >= sqMinY && y < sqMaxY)
            for x in 0..<width {
                let offset = (y * width + x) * 3
                if inSquareY && x >= sqMinX && x < sqMaxX {
                    rgb[offset] = 255
                    rgb[offset + 1] = 0
                    rgb[offset + 2] = 0
                } else {
                    rgb[offset] = 0
                    rgb[offset + 1] = 0
                    rgb[offset + 2] = 255
                }
            }
        }
        return rgb
    }
}
