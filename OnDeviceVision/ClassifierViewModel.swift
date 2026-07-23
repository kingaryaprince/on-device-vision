//  ClassifierViewModel.swift
//  Runs the Core ML model on-device via Vision. No network calls anywhere in
//  this file by design -- that is the whole privacy point of the project.

import SwiftUI
import Vision
import CoreML
import CoreImage

@MainActor
final class ClassifierViewModel: ObservableObject {
    struct Prediction: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Double
    }

    @Published var predictions: [Prediction] = []
    @Published var latencyMs: Double = 0
    @Published var errorText: String?

    private lazy var request: VNCoreMLRequest? = makeRequest()

    private func makeRequest() -> VNCoreMLRequest? {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all              // let iOS pick the Neural Engine
            // NOTE: `MobileNetV2Int8` is the class Xcode auto-generates once you
            // drag MobileNetV2Int8.mlpackage into the project. Rename if you used a
            // different --out name in convert_model.py.
            let coreModel = try MobileNetV2Int8(configuration: config).model
            let vnModel = try VNCoreMLModel(for: coreModel)
            let req = VNCoreMLRequest(completionHandler: handleResults)
            req.imageCropAndScaleOption = .centerCrop
            return req
        } catch {
            self.errorText = "Model load failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Feed one frame (called from the camera controller). Fully on-device.
    func classify(_ pixelBuffer: CVPixelBuffer) {
        guard let request else { return }
        let start = CFAbsoluteTimeGetCurrent()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        do {
            try handler.perform([request])
            self.latencyMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        } catch {
            self.errorText = "Inference failed: \(error.localizedDescription)"
        }
    }

    private func handleResults(_ request: VNRequest, _ error: Error?) {
        guard let results = request.results as? [VNClassificationObservation] else { return }
        let top3 = results.prefix(3).map {
            Prediction(label: $0.identifier, confidence: Double($0.confidence))
        }
        Task { @MainActor in self.predictions = top3 }
    }
}
