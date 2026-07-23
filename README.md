# On-Device Vision Classifier

A fully **offline** iOS app that classifies the live camera feed **entirely on-device**
using Core ML — no network calls, nothing leaves the phone. Works in airplane mode.

Built to demonstrate on-device ML with a privacy-first design: a PyTorch model
converted and **INT8-quantized** with `coremltools`, then run through the Vision
framework on the Neural Engine.

## Why this exists
Server-side inference is easy; making a model measurably correct, small, and fast
*on the device in front of you* is the harder and more interesting problem. This
project converts a model, quantizes it, verifies the quantized model still agrees
with the original, and measures the size / latency / accuracy tradeoff.

## Architecture
```
MobileNetV2 (PyTorch)
    │  torch.jit.trace
    ▼
coremltools.convert  ──►  INT8 linear_quantize_weights  ──►  MobileNetV2Int8.mlpackage
    │                                                              │
    │  parity check (PyTorch vs Core ML top-1)                     │  drag into Xcode
    ▼                                                              ▼
convert_model.py (host)                              Vision + Core ML on iPhone (offline)
```

## Setup
### 1. Convert + quantize the model (host, macOS)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python convert_model.py --image sample.jpg
```
This writes `OnDeviceVision/MobileNetV2Int8.mlpackage` and prints a PyTorch-vs-CoreML
parity check.

### 2. Build the app
1. Create a new iOS App in Xcode (SwiftUI), name it `OnDeviceVision`.
2. Add the four `.swift` files from `OnDeviceVision/`.
3. Drag `MobileNetV2Int8.mlpackage` into the project (Xcode auto-generates the
   `MobileNetV2Int8` Swift class the view model references).
4. Add `NSCameraUsageDescription` to Info.plist ("Used to classify the camera feed on-device").
5. Build to a **real device** (camera + Neural Engine aren't in the simulator).

## Results

Measured from an actual `convert_model.py` run (MobileNetV2, ImageNet weights):

| Model            | Size (.mlpackage) | Sample parity vs PyTorch | On-host latency¹ | On-device latency² |
|------------------|-------------------|--------------------------|------------------|--------------------|
| MobileNetV2 fp16 | 7.07 MB           | —                        | —                | TODO               |
| MobileNetV2 INT8 | 3.69 MB           | top-1 agrees ("Samoyed") | 11.2 ms          | TODO               |

**INT8 vs fp16 baseline: 1.92× smaller** (7.07 → 3.69 MB), top-1 prediction unchanged
on the sample image. The reduction is ~2× rather than ~4× because coremltools already
stores fp16 (16-bit) weights by default, so INT8 halves an already-halved baseline.

¹ On-host = this Mac via the coremltools runtime; representative, not the shipping target.
² On-device latency requires building in Xcode and deploying to a physical iPhone
(camera + Neural Engine are not available in the simulator) — fill from a real device run.

> Note: INT8 weight quantization emitted divide-by-zero warnings on a few zero-range
> weight tensors; top-1 parity still held on the test image. Validate on a larger set
> before trusting it broadly — that verification is exactly the interesting part.

## Privacy note
There is no networking code in this repo. All inference runs locally; the camera
buffer never leaves the process. That is the design, not an afterthought.

## Possible extensions
- Swap MobileNetV2 for a tiny object detector (YOLO) and draw boxes with Vision.
- Compare `.cpuOnly` vs `.all` compute units for the latency table.
- Add on-device fine-tuning with Core ML `MLUpdateTask`.
