// Runs Apple Vision 2D and 3D human body pose on every JPEG in a folder and
// writes one JSON object per image (JSON lines). Images are read in stored
// pixel orientation (.up) so coordinates line up with annotations made on the
// raw pixels. All normalized image coordinates are written with a TOP-LEFT
// origin (Vision's lower-left origin is flipped here).
// 3D runs twice: on the full image, and with the region of interest set to each
// 2D person box (padded), the way the App sets ROI = person box.
// Output carries joints and a 64-bit difference hash per image only, never pixels.
import CoreGraphics
import Foundation
import ImageIO
import Vision
import simd

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: BodyPoseExtract <image-dir> <out.jsonl>\n".data(using: .utf8)!)
    exit(2)
}
let imageDir = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])

let joints2D: [(String, VNHumanBodyPoseObservation.JointName)] = [
    ("nose", .nose), ("leftEye", .leftEye), ("rightEye", .rightEye),
    ("leftEar", .leftEar), ("rightEar", .rightEar), ("neck", .neck),
    ("leftShoulder", .leftShoulder), ("rightShoulder", .rightShoulder),
    ("leftElbow", .leftElbow), ("rightElbow", .rightElbow),
    ("leftWrist", .leftWrist), ("rightWrist", .rightWrist), ("root", .root),
    ("leftHip", .leftHip), ("rightHip", .rightHip), ("leftKnee", .leftKnee),
    ("rightKnee", .rightKnee), ("leftAnkle", .leftAnkle), ("rightAnkle", .rightAnkle),
]

let joints3D: [(String, VNHumanBodyPose3DObservation.JointName)] = [
    ("topHead", .topHead), ("centerHead", .centerHead), ("centerShoulder", .centerShoulder),
    ("leftShoulder", .leftShoulder), ("rightShoulder", .rightShoulder),
    ("leftElbow", .leftElbow), ("rightElbow", .rightElbow),
    ("leftWrist", .leftWrist), ("rightWrist", .rightWrist), ("spine", .spine),
    ("root", .root), ("leftHip", .leftHip), ("rightHip", .rightHip),
    ("leftKnee", .leftKnee), ("rightKnee", .rightKnee),
    ("leftAnkle", .leftAnkle), ("rightAnkle", .rightAnkle),
]

func r4(_ v: Float) -> Double { (Double(v) * 10000).rounded() / 10000 }
func r4(_ v: Double) -> Double { (v * 10000).rounded() / 10000 }
func translation(_ m: simd_float4x4) -> [Double] { [r4(m.columns.3.x), r4(m.columns.3.y), r4(m.columns.3.z)] }
func matrix(_ m: simd_float4x4) -> [Double] {
    // column-major, 16 values
    [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [r4($0.x), r4($0.y), r4($0.z), r4($0.w)] }
}

func dHash(_ image: CGImage) -> String {
    let w = 9, h = 8
    var pixels = [UInt8](repeating: 0, count: w * h)
    let space = CGColorSpaceCreateDeviceGray()
    guard let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w, space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue)
    else { return "" }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    var bits: UInt64 = 0
    for y in 0..<h { for x in 0..<(w - 1) {
        bits <<= 1
        if pixels[y * w + x] > pixels[y * w + x + 1] { bits |= 1 }
    } }
    return String(format: "%016llx", bits)
}

func describe2D(_ obs: VNHumanBodyPoseObservation) -> [String: Any] {
    var js: [String: Any] = [:]
    var xs: [Double] = [], ys: [Double] = []
    for (name, key) in joints2D {
        guard let p = try? obs.recognizedPoint(key), p.confidence > 0 else { continue }
        let x = Double(p.location.x), y = 1 - Double(p.location.y)
        js[name] = [r4(x), r4(y), r4(Double(p.confidence))]
        if p.confidence >= 0.1 { xs.append(x); ys.append(y) }
    }
    var out: [String: Any] = ["conf": r4(Double(obs.confidence)), "joints": js]
    if let x0 = xs.min(), let x1 = xs.max(), let y0 = ys.min(), let y1 = ys.max() {
        out["box"] = [r4(x0), r4(y0), r4(x1 - x0), r4(y1 - y0)]  // top-left x, y, w, h
    }
    return out
}

func describe3D(_ obs: VNHumanBodyPose3DObservation) -> [String: Any] {
    var js: [String: Any] = [:]
    for (name, key) in joints3D {
        var j: [String: Any] = [:]
        if let p = try? obs.recognizedPoint(key) { j["model"] = translation(p.position) }
        if let c = try? obs.cameraRelativePosition(key) { j["cam"] = translation(c) }
        if let q = try? obs.pointInImage(key) { j["img"] = [r4(Double(q.x)), r4(1 - Double(q.y))] }
        if !j.isEmpty { js[name] = j }
    }
    let est: String
    switch obs.heightEstimation {
    case .measured: est = "measured"
    case .reference: est = "reference"
    @unknown default: est = "unknown"
    }
    return [
        "conf": r4(Double(obs.confidence)),
        "bodyHeight": r4(obs.bodyHeight),
        "heightEstimation": est,
        "cameraOrigin": matrix(obs.cameraOriginMatrix),
        "joints": js,
    ]
}

func flippedROI(_ box: [Double], pad: Double) -> CGRect {
    // top-left box -> Vision lower-left ROI, padded and clamped to [0,1]
    let x0 = max(0, box[0] - box[2] * pad), x1 = min(1, box[0] + box[2] * (1 + pad))
    let y0 = max(0, box[1] - box[3] * pad), y1 = min(1, box[1] + box[3] * (1 + pad))
    return CGRect(x: x0, y: 1 - y1, width: x1 - x0, height: y1 - y0)
}

func run3D(_ image: CGImage, roi: CGRect?) -> [String: Any] {
    let request = VNDetectHumanBodyPose3DRequest()
    if let roi { request.regionOfInterest = roi }
    let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
    do {
        try handler.perform([request])
        let results = request.results ?? []
        return ["n": results.count, "obs": results.map(describe3D)]
    } catch {
        return ["error": String(describing: error)]
    }
}

let files = (try? FileManager.default.contentsOfDirectory(at: imageDir, includingPropertiesForKeys: nil)) ?? []
let images = files.filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
FileManager.default.createFile(atPath: outURL.path, contents: nil)
guard let out = try? FileHandle(forWritingTo: outURL) else { exit(3) }

var done = 0
for url in images {
    autoreleasepool {
        var row: [String: Any] = ["id": url.deletingPathExtension().lastPathComponent]
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            row["error"] = "decode"
            out.write(try! JSONSerialization.data(withJSONObject: row) + Data("\n".utf8))
            return
        }
        row["w"] = image.width
        row["h"] = image.height
        row["dhash"] = dHash(image)
        let t0 = Date()
        let req2 = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        do {
            try handler.perform([req2])
            let obs = (req2.results ?? []).map(describe2D)
            row["p2d"] = obs
            row["t2d"] = r4(Date().timeIntervalSince(t0))
            let t1 = Date()
            row["p3dFull"] = run3D(image, roi: nil)
            // ROI per 2D person (largest four by box area)
            let ranked = obs.enumerated().compactMap { (i, o) -> (Int, [Double])? in
                guard let b = o["box"] as? [Double] else { return nil }
                return (i, b)
            }.sorted { $0.1[2] * $0.1[3] > $1.1[2] * $1.1[3] }.prefix(4)
            row["p3dRoi"] = ranked.map { (i, b) -> [String: Any] in
                var r = run3D(image, roi: flippedROI(b, pad: 0.15))
                r["p2dIndex"] = i
                return r
            }
            row["t3d"] = r4(Date().timeIntervalSince(t1))
        } catch {
            row["error2d"] = String(describing: error)
        }
        out.write(try! JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]) + Data("\n".utf8))
        done += 1
        if done % 100 == 0 { print("processed \(done)/\(images.count)") }
    }
}
try? out.close()
print("BODYPOSE_DONE \(done)/\(images.count)")
