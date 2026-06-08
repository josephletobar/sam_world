import torch
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path

# load DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()

# Folder containing images

# image_dir = Path(r"C:\Windows\System32\forest_frames\zed2i_left_images")

image_dir = Path(r"C:\Users\jleto\Downloads\rgbd_dataset_freiburg1_xyz\rgbd_dataset_freiburg1_xyz\rgb")
image_dir = Path(r"D:\forest_frames\zed2i_left_images")

img_paths = sorted(
    p for p in image_dir.glob("*.jpeg")
    if not p.name.startswith("._")
)

for img_path in img_paths:

    # load image
    image = Image.open(img_path)
    image = image.resize((768, 768))

    # forward pass
    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # remove CLS token → patch tokens
    patches = outputs.last_hidden_state[:, 1:, :].squeeze(0)

    # normalize for cosine clustering
    patches = torch.nn.functional.normalize(patches, dim=1)

    # DBSCAN clustering
    labels = DBSCAN(
        eps=0.3,
        min_samples=3,
        metric="cosine"
    ).fit_predict(patches.cpu().numpy())

    print("clusters:", np.unique(labels))

    # reshape to patch grid (16x16)
    cluster_map = labels.reshape(16, 16)

    # load original image for overlay
    img_bgr = cv2.imread(str(img_path))
    img_bgr = cv2.resize(img_bgr, (512, 512))

    # upscale cluster map to image size
    heat = cluster_map.astype(np.float32)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-6)

    heat = cv2.resize(
        heat,
        (512, 512),
        interpolation=cv2.INTER_NEAREST
    )

    # colorize
    heat_color = cv2.applyColorMap(
        (heat * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # overlay on image
    overlay = cv2.addWeighted(img_bgr, 0.65, heat_color, 0.35, 0)

    cv2.imshow("DINOv2 Cluster Overlay", overlay)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        cv2.waitKey(0)

cv2.destroyAllWindows()