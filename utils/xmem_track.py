from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import numpy as np
from pathlib import Path

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

import sys
sys.path.append(r"C:\Users\jletobar3\Projects\XMem")
from inference.inference_core import InferenceCore
from model.network import XMem
from inference.interact.interactive_utils import image_to_torch


config = {
    "mem_every": 20,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "max_long_term_elements": 1000,
    "num_prototypes": 128,
    "top_k": 30,
    "num_objects": 1,
}

network = XMem(config=config, model_path="XMem.pth").to(device).eval()

inference = InferenceCore(network=network, config=config)
inference.set_all_labels([1])

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

image_dir = Path(r"D:\kab3_data\rgb")
image_paths = sorted(image_dir.glob("*"))

# Disable gradient computation for inference
torch.set_grad_enabled(False)

# ---------- FIRST FRAME ----------
first_img = cv2.imread(str(image_paths[0]))
first_img = cv2.resize(first_img, (640, 480))

with torch.no_grad():
    predictor.set_image(first_img)
    results = predictor(text=["black car"])

    annotated = first_img.copy()

    if len(results):
        r = results[0]

        if r.masks is not None:
            annotated = r.plot()

            sam3_mask = r.masks.data[0]
            sam3_mask = sam3_mask.float().unsqueeze(0)
            print(f"First frame - SAM3 mask shape: {sam3_mask.shape}")

            img_torch, _ = image_to_torch(first_img, device)    
            inference.step(img_torch, sam3_mask)

# Clean up SAM3 predictor state
if hasattr(predictor, 'predictor') and hasattr(predictor.predictor, 'features'):
    predictor.predictor.features = None
if device.type == 'cuda':
    torch.cuda.empty_cache()

cv2.imshow("Segmentation", annotated)

# Wait for space key to start XMem tracking
print("Press SPACE to start XMem tracking...")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 32:  # Space key
        break

prev_mask = sam3_mask  # Store the initial mask for future use

# ---------- REMAINING FRAMES ----------
for frame_idx, img_path in enumerate(image_paths[1:], start=1):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img = cv2.resize(img, (640, 480))
    
    with torch.no_grad():
        img_torch, _ = image_to_torch(img, device)    
        mask = inference.step(img_torch)
    
    print(f"Frame {frame_idx} - Output mask shape: {mask.shape}")

    # Get the object mask (index 1, skipping background at index 0)
    mask_prob = mask[1].detach().cpu().numpy()  # Range: 0-1
    mask_prob[mask_prob < 0.9] = 0.0
    segmented_rgb = img * ((mask_prob > 0.5)[..., None])

    # Create blue highlight overlay on original image
    annotated = img.copy()
    
    # Threshold the mask for cleaner visualization
    mask_binary = (mask_prob > 0.5).astype(float)
    
    # Create blue color (BGR format: Blue=255, Green=0, Red=0)
    blue_overlay = np.zeros_like(img)
    blue_overlay[:, :, 0] = 255  # Blue channel
    
    # Blend: where mask is high, show more blue
    alpha = mask_prob * 0.6  # 60% opacity at mask=1
    for c in range(3):
        annotated[:, :, c] = (img[:, :, c] * (1 - alpha) + blue_overlay[:, :, c] * alpha).astype(np.uint8)

    cv2.imshow("Segmentation", annotated)

    # Periodic memory cleanup
    if frame_idx % 10 == 0 and device.type == 'cuda':
        torch.cuda.empty_cache()

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
if device.type == 'cuda':
    torch.cuda.empty_cache()
