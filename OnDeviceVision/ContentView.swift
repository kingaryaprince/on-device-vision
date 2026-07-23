//  ContentView.swift
//  On-device image classifier UI: live camera + top-3 predictions + latency.

import SwiftUI

@main
struct OnDeviceVisionApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var vm = ClassifierViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraView { buffer in vm.classify(buffer) }
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 8) {
                if let err = vm.errorText {
                    Text(err).foregroundStyle(.red).font(.footnote)
                }
                ForEach(vm.predictions) { p in
                    HStack {
                        Text(p.label.capitalized)
                            .font(.headline)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int(p.confidence * 100))%")
                            .font(.headline.monospacedDigit())
                    }
                }
                Text(String(format: "on-device • %.0f ms/frame • offline", vm.latencyMs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding()
        }
    }
}
