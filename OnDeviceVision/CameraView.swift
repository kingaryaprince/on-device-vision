//  CameraView.swift
//  Minimal AVCaptureSession that pipes frames to the view model. This is the
//  "wire up the live camera" step -- keep it small; the ML is the point.

import SwiftUI
import AVFoundation

struct CameraView: UIViewControllerRepresentable {
    let onFrame: (CVPixelBuffer) -> Void

    func makeUIViewController(context: Context) -> CameraController {
        let vc = CameraController()
        vc.onFrame = onFrame
        return vc
    }
    func updateUIViewController(_ vc: CameraController, context: Context) {}
}

final class CameraController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onFrame: ((CVPixelBuffer) -> Void)?
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "camera.frames")
    private var lastRun = Date.distantPast

    override func viewDidLoad() {
        super.viewDidLoad()
        configure()
    }

    private func configure() {
        session.sessionPreset = .high
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        preview.frame = view.bounds
        view.layer.addSublayer(preview)

        Task.detached { [session] in session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Throttle to ~5 fps so we are not classifying every single frame.
        guard Date().timeIntervalSince(lastRun) > 0.2,
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastRun = Date()
        onFrame?(buffer)
    }
}
