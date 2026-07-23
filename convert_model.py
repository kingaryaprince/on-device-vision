"""
convert_model.py — PyTorch -> Core ML, with INT8 quantization + parity check.

Produces MobileNetV2Int8.mlpackage that the iOS app loads and runs fully
on-device. The point of this script (and the resume bullet it backs) is not
"I called a converter" -- it is that you convert, quantize, and then *verify*
the quantized model still agrees with the original, and measure the
size/latency vs. accuracy tradeoff.

Usage:
    python -m venv .venv && source .venv/bin/activate
    pip install torch torchvision coremltools pillow numpy
    python convert_model.py --image sample.jpg

Requires macOS for the Core ML prediction / parity step (coremltools runs
the CoreML model via the OS runtime).
"""

import argparse
import time
import urllib.request
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import coremltools as ct
from coremltools.optimize.coreml import (
    linear_quantize_weights,
    OpLinearQuantizerConfig,
    OptimizationConfig,
)

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
# A stable sample image so `python convert_model.py` runs with zero manual setup.
SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
)


def ensure_sample_image(path="sample.jpg"):
    """Download a sample image once if it isn't already present."""
    import os
    if not os.path.exists(path):
        try:
            print(f"      fetching sample image -> {path}")
            urllib.request.urlretrieve(SAMPLE_IMAGE_URL, path)
        except Exception as e:
            print(f"      could not fetch sample image ({e}); using random input")
            return None
    return path


def load_labels():
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as f:
            return [line.strip().decode() for line in f.readlines()]
    except Exception:
        # Offline fallback: numeric labels still let the parity check run.
        return [str(i) for i in range(1000)]


def preprocess(image_path):
    """Standard ImageNet preprocessing -> [1,3,224,224] tensor."""
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0), img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="Image for the parity check. If omitted, a sample is fetched automatically.")
    ap.add_argument("--out", default="OnDeviceVision/MobileNetV2Int8.mlpackage")
    args = ap.parse_args()

    # Zero-setup default: grab a sample image if the user didn't pass one.
    if args.image is None:
        args.image = ensure_sample_image()

    labels = load_labels()

    # 1) Load a small pretrained model (swap for a tiny YOLO later if you want detection)
    print("[1/5] Loading MobileNetV2 (pretrained)...")
    torch_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()

    # 2) Prepare input
    if args.image:
        x, pil_img = preprocess(args.image)
    else:
        print("      no --image given, using random input for parity check")
        x = torch.rand(1, 3, 224, 224)

    # 3) Trace + convert to Core ML with an image input type (so Swift/Vision can feed a CGImage)
    print("[2/5] Tracing + converting to Core ML (fp16)...")
    traced = torch.jit.trace(torch_model, x)
    scale = 1.0 / (255.0 * 0.226)          # folded ImageNet std (approx, single scale)
    bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    mlmodel_fp = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=x.shape, scale=scale, bias=bias)],
        classifier_config=ct.ClassifierConfig(labels),
        compute_units=ct.ComputeUnit.CPU_AND_NE,   # prefer the Neural Engine on-device
        minimum_deployment_target=ct.target.iOS16,
    )

    # 4) INT8 weight quantization (this is the part you can speak to from the Parallax work)
    print("[3/5] INT8 quantizing weights...")
    q_config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    mlmodel_int8 = linear_quantize_weights(mlmodel_fp, config=q_config)
    mlmodel_int8.save(args.out)
    print(f"      saved -> {args.out}")

    # 5) Parity check: does the quantized CoreML model still agree with PyTorch?
    print("[4/5] Parity check (PyTorch vs quantized Core ML)...")
    with torch.no_grad():
        torch_logits = torch_model(x)
    torch_top = int(torch.argmax(torch_logits, dim=1))

    if args.image:
        t0 = time.perf_counter()
        pred = mlmodel_int8.predict({"image": pil_img.resize((224, 224))})
        latency_ms = (time.perf_counter() - t0) * 1000
        # classifier output is a label->prob dict under the model's output name
        prob_key = [k for k, v in pred.items() if isinstance(v, dict)]
        coreml_label = pred.get("classLabel") or (
            max(pred[prob_key[0]].items(), key=lambda kv: kv[1])[0] if prob_key else "?"
        )
        print(f"      PyTorch top-1:  {labels[torch_top]}")
        print(f"      Core ML top-1:  {coreml_label}")
        print(f"      agree: {labels[torch_top] == coreml_label}")
        print(f"      Core ML on-host latency: {latency_ms:.1f} ms")
    else:
        print(f"      PyTorch top-1 index: {torch_top} (pass --image for a full label parity check)")

    print("[5/5] Done. Drag the .mlpackage into the Xcode project and build to device.")
    print("      TODO(you): fill the size/latency/accuracy table in README from real runs.")


if __name__ == "__main__":
    main()
