import SwiftUI
import llama

@main
struct QwenProbeHostApp: App {
    private let llamaBuildInfo: String = {
        if let ver = llama_version() {
            return String(cString: ver)
        }
        return "unknown"
    }()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 12) {
                Text("Qwen iPhone Probe Host")
                    .font(.headline)
                Text("llama.cpp \(llamaBuildInfo)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
        }
    }
}
