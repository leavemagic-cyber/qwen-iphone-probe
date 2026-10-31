import Darwin
import Foundation
import UIKit
import XCTest

private struct QwenProbePayload: Encodable {
    let variant: String
    let device_model_identifier: String
    let ios_version: String
    let os_proc_available_memory_before_load_bytes: UInt64
    let phys_footprint_after_model_load_bytes: UInt64
    let peak_phys_footprint_bytes: UInt64
    let model_load_seconds: Double
    let image_encode_seconds: Double
    let image_encode_warm_seconds: Double
    let time_to_first_token_seconds: Double
    let generated_tokens_per_second: Double
    let generated_tokens: Int32
    let generated_text: String
    let error: String?
    let model_url: String
    let model_bytes: UInt64
    let mmproj_url: String
    let mmproj_bytes: UInt64

    enum CodingKeys: String, CodingKey {
        case variant
        case device_model_identifier
        case ios_version
        case os_proc_available_memory_before_load_bytes
        case phys_footprint_after_model_load_bytes
        case peak_phys_footprint_bytes
        case model_load_seconds
        case image_encode_seconds
        case image_encode_warm_seconds
        case time_to_first_token_seconds
        case generated_tokens_per_second
        case generated_tokens
        case generated_text
        case error
        case model_url
        case model_bytes
        case mmproj_url
        case mmproj_bytes
    }

    private static func clean(_ value: Double) -> Double {
        return value.isFinite ? (value * 1_000_000.0).rounded() / 1_000_000.0 : 0.0
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(variant, forKey: .variant)
        try c.encode(device_model_identifier, forKey: .device_model_identifier)
        try c.encode(ios_version, forKey: .ios_version)
        try c.encode(os_proc_available_memory_before_load_bytes, forKey: .os_proc_available_memory_before_load_bytes)
        try c.encode(phys_footprint_after_model_load_bytes, forKey: .phys_footprint_after_model_load_bytes)
        try c.encode(peak_phys_footprint_bytes, forKey: .peak_phys_footprint_bytes)
        try c.encode(Self.clean(model_load_seconds), forKey: .model_load_seconds)
        try c.encode(Self.clean(image_encode_seconds), forKey: .image_encode_seconds)
        try c.encode(Self.clean(image_encode_warm_seconds), forKey: .image_encode_warm_seconds)
        try c.encode(Self.clean(time_to_first_token_seconds), forKey: .time_to_first_token_seconds)
        try c.encode(Self.clean(generated_tokens_per_second), forKey: .generated_tokens_per_second)
        try c.encode(generated_tokens, forKey: .generated_tokens)
        try c.encode(generated_text, forKey: .generated_text)
        try c.encode(error, forKey: .error)
        try c.encode(model_url, forKey: .model_url)
        try c.encode(model_bytes, forKey: .model_bytes)
        try c.encode(mmproj_url, forKey: .mmproj_url)
        try c.encode(mmproj_bytes, forKey: .mmproj_bytes)
    }
}

private final class ProbeSharedState {
    private let lock = NSLock()
    var payload: QwenProbePayload?
    var modelBytes: UInt64 = 0
    var mmprojBytes: UInt64 = 0
    var availBeforeLoad: UInt64 = 0

    func updateDownloadProgress(modelBytes: UInt64? = nil, mmprojBytes: UInt64? = nil, availBeforeLoad: UInt64? = nil) {
        lock.lock()
        if let m = modelBytes { self.modelBytes = m }
        if let mm = mmprojBytes { self.mmprojBytes = mm }
        if let a = availBeforeLoad { self.availBeforeLoad = a }
        lock.unlock()
    }

    func setPayload(_ value: QwenProbePayload) {
        lock.lock()
        self.payload = value
        lock.unlock()
    }

    func snapshot() -> (payload: QwenProbePayload?, modelBytes: UInt64, mmprojBytes: UInt64, availBeforeLoad: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        return (payload, modelBytes, mmprojBytes, availBeforeLoad)
    }
}

@objc(QwenProbeTests)
final class QwenProbeTests: XCTestCase {
    private static let modelURLString =
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    private static let modelFilename = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    private static let expectedModelBytes: UInt64 = 1_107_409_952

    private static let mmprojURLString =
        "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let mmprojFilename = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    private static let expectedMmprojBytes: UInt64 = 445_053_216

    private static let promptText =
        "What colours and shapes are in this image? Answer in one sentence."
    private static let maxTokens: Int32 = 64
    private static let internalTimeoutSeconds: TimeInterval = 2400.0

    // All methods keep the A13 crash fix (output head on CPU, op_offload off) and vary only
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
        for look in 1...10 {
            let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
            let start = Date()
            runProbeVariant(variant: String(format: "app_profile_look_%02d", look),
                            options: Self.a13Options(clipCpu: 0, clipThreads: 0), width: 576, height: 768)
            Self.logLine(
                "QWENPROBE_LOOK look=\(look) wall_seconds=\(String(format: "%.3f", Date().timeIntervalSince(start))) " +
                "thermal_before=\(thermalBefore) thermal_after=\(ProcessInfo.processInfo.thermalState.rawValue)"
            )
        }
    }

    private func runProbeVariant(variant: String, options: qwen_probe_options, width: Int, height: Int) {
        continueAfterFailure = false
        executionTimeAllowance = 2400

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

        let deviceModel = Self.readDeviceModelIdentifier()
        let initialAvail = qwen_probe_available_memory()
        qwen_probe_record_live_footprint()

        Self.logLine(
            "[QWENPROBE] variant=\(variant) step=test_start device=\(deviceModel) ios=\(iosVersion) " +
            "avail_bytes=\(initialAvail) phys_footprint_bytes=\(qwen_probe_phys_footprint())"
        )

        let state = ProbeSharedState()
        state.updateDownloadProgress(availBeforeLoad: initialAvail)

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
                    let failPayload = QwenProbePayload(
                        variant: variant,
                        device_model_identifier: deviceModel,
                        ios_version: iosVersion,
                        os_proc_available_memory_before_load_bytes: qwen_probe_available_memory(),
                        phys_footprint_after_model_load_bytes: 0,
                        peak_phys_footprint_bytes: qwen_probe_live_peak_footprint(),
                        model_load_seconds: 0.0,
                        image_encode_seconds: 0.0,
                        image_encode_warm_seconds: 0.0,
                        time_to_first_token_seconds: 0.0,
                        generated_tokens_per_second: 0.0,
                        generated_tokens: 0,
                        generated_text: "",
                        error: "Failed to prepare Caches directory: \(error.localizedDescription)",
                        model_url: Self.modelURLString,
                        model_bytes: 0,
                        mmproj_url: Self.mmprojURLString,
                        mmproj_bytes: 0
                    )
                    state.setPayload(failPayload)
                    return
                }

                let modelDest = cacheDir.appendingPathComponent(Self.modelFilename)
                let mmprojDest = cacheDir.appendingPathComponent(Self.mmprojFilename)

                var cachedModelBytes: UInt64 = 0
                var cachedMmprojBytes: UInt64 = 0

                do {
                    cachedModelBytes = try Self.ensureCachedGGUF(
                        variant: variant,
                        label: "model",
                        urlString: Self.modelURLString,
                        expectedBytes: Self.expectedModelBytes,
                        destination: modelDest
                    )
                    state.updateDownloadProgress(modelBytes: cachedModelBytes)
                    qwen_probe_record_live_footprint()

                    cachedMmprojBytes = try Self.ensureCachedGGUF(
                        variant: variant,
                        label: "mmproj",
                        urlString: Self.mmprojURLString,
                        expectedBytes: Self.expectedMmprojBytes,
                        destination: mmprojDest
                    )
                    state.updateDownloadProgress(mmprojBytes: cachedMmprojBytes)
                    qwen_probe_record_live_footprint()
                } catch {
                    let availNow = qwen_probe_available_memory()
                    let peakNow = qwen_probe_live_peak_footprint()
                    let failPayload = QwenProbePayload(
                        variant: variant,
                        device_model_identifier: deviceModel,
                        ios_version: iosVersion,
                        os_proc_available_memory_before_load_bytes: availNow,
                        phys_footprint_after_model_load_bytes: 0,
                        peak_phys_footprint_bytes: peakNow,
                        model_load_seconds: 0.0,
                        image_encode_seconds: 0.0,
                        image_encode_warm_seconds: 0.0,
                        time_to_first_token_seconds: 0.0,
                        generated_tokens_per_second: 0.0,
                        generated_tokens: 0,
                        generated_text: "",
                        error: "Download failed: \(error.localizedDescription)",
                        model_url: Self.modelURLString,
                        model_bytes: cachedModelBytes,
                        mmproj_url: Self.mmprojURLString,
                        mmproj_bytes: cachedMmprojBytes
                    )
                    state.setPayload(failPayload)
                    return
                }

                let rgbBytes = Self.makeSyntheticRGBImage(width: width, height: height)
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
                var optsCopy = options

                Self.logLine(
                    "[QWENPROBE] variant=\(variant) step=starting_engine width=\(width) " +
                    "height=\(height) max_tokens=\(Self.maxTokens) " +
                    "phys_footprint_bytes=\(qwen_probe_phys_footprint())"
                )

                let rc: Int32 = withUnsafePointer(to: &optsCopy) { optsPtr in
                    rgbBytes.withUnsafeBufferPointer { rgbPtr in
                        textBuf.withUnsafeMutableBufferPointer { textPtr in
                            errBuf.withUnsafeMutableBufferPointer { errPtr in
                                variant.withCString { variantCStr in
                                    modelPath.withCString { modelCStr in
                                        mmprojPath.withCString { mmprojCStr in
                                            Self.promptText.withCString { promptCStr in
                                                qwen_probe_run(
                                                    variantCStr,
                                                    optsPtr,
                                                    modelCStr,
                                                    mmprojCStr,
                                                    rgbPtr.baseAddress,
                                                    UInt32(width),
                                                    UInt32(height),
                                                    promptCStr,
                                                    Self.maxTokens,
                                                    metricsPtr,
                                                    textPtr.baseAddress,
                                                    textPtr.count,
                                                    errPtr.baseAddress,
                                                    errPtr.count
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

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
                    errorMessage = engineError.isEmpty ? "qwen_probe_run failed with code \(rc)" : engineError
                } else if generatedText.isEmpty {
                    errorMessage = "Model loaded and ran without error code, but generated_text was empty"
                } else {
                    errorMessage = nil
                }

                let peakFootprint = max(metrics.peak_phys_footprint_bytes, qwen_probe_live_peak_footprint())
                let availRecorded = metrics.avail_mem_before_load_bytes > 0
                    ? metrics.avail_mem_before_load_bytes
                    : availBefore

                let finalPayload = QwenProbePayload(
                    variant: variant,
                    device_model_identifier: deviceModel,
                    ios_version: iosVersion,
                    os_proc_available_memory_before_load_bytes: availRecorded,
                    phys_footprint_after_model_load_bytes: metrics.phys_footprint_after_model_load_bytes,
                    peak_phys_footprint_bytes: peakFootprint,
                    model_load_seconds: metrics.model_load_seconds,
                    image_encode_seconds: metrics.image_encode_seconds,
                    image_encode_warm_seconds: metrics.image_encode_warm_seconds,
                    time_to_first_token_seconds: metrics.time_to_first_token_seconds,
                    generated_tokens_per_second: metrics.generated_tokens_per_second,
                    generated_tokens: metrics.generated_tokens,
                    generated_text: generatedText,
                    error: errorMessage,
                    model_url: Self.modelURLString,
                    model_bytes: cachedModelBytes,
                    mmproj_url: Self.mmprojURLString,
                    mmproj_bytes: cachedMmprojBytes
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
            finalPayload = QwenProbePayload(
                variant: variant,
                device_model_identifier: deviceModel,
                ios_version: iosVersion,
                os_proc_available_memory_before_load_bytes: snap.availBeforeLoad > 0
                    ? snap.availBeforeLoad
                    : qwen_probe_available_memory(),
                phys_footprint_after_model_load_bytes: 0,
                peak_phys_footprint_bytes: qwen_probe_live_peak_footprint(),
                model_load_seconds: 0.0,
                image_encode_seconds: 0.0,
                image_encode_warm_seconds: 0.0,
                time_to_first_token_seconds: 0.0,
                generated_tokens_per_second: 0.0,
                generated_tokens: 0,
                generated_text: "",
                error: timeoutReason,
                model_url: Self.modelURLString,
                model_bytes: snap.modelBytes,
                mmproj_url: Self.mmprojURLString,
                mmproj_bytes: snap.mmprojBytes
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
        destination: URL
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
            destination: destination
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
        return "{\"variant\":\"\(payload.variant)\"," +
            "\"device_model_identifier\":\"\(payload.device_model_identifier)\"," +
            "\"ios_version\":\"\(payload.ios_version)\"," +
            "\"os_proc_available_memory_before_load_bytes\":\(payload.os_proc_available_memory_before_load_bytes)," +
            "\"peak_phys_footprint_bytes\":\(payload.peak_phys_footprint_bytes)," +
            "\"generated_text\":\"\",\"error\":\"\(escapedError)\"}"
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
        destination: URL
    ) throws -> UInt64 {
        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "QwenProbeDownload",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid \(label) URL: \(urlString)"]
            )
        }

        let partURL = destination.appendingPathExtension("part")
        let maxAttempts = 3
        var lastError: Error?

        for attempt in 1...maxAttempts {
            try? FileManager.default.removeItem(at: partURL)
            logLine(
                "[QWENPROBE] variant=\(variant) step=download_start label=\(label) " +
                "attempt=\(attempt)/\(maxAttempts) url=\(urlString) expected_bytes=\(expectedBytes) " +
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
                            userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode) for \(urlString)"]
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
                logLine("[QWENPROBE] variant=\(variant) Download \(label) attempt=\(attempt) failed: \(err.localizedDescription)")
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
                logLine("[QWENPROBE] variant=\(variant) Validation \(label) attempt=\(attempt) failed: \(error.localizedDescription)")
                if attempt < maxAttempts {
                    Thread.sleep(forTimeInterval: TimeInterval(attempt * 3))
                }
            }
        }

        throw lastError ?? NSError(
            domain: "QwenProbeDownload",
            code: -99,
            userInfo: [NSLocalizedDescriptionKey: "Exhausted retries downloading \(label) from \(urlString)"]
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
