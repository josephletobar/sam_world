import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# image_dir = Path(r"C:\Windows\System32\forest_frames\zed2i_left_images")

image_dir = Path(
    r"C:\Users\jleto\Downloads\rgbd_dataset_freiburg1_xyz\rgbd_dataset_freiburg1_xyz\rgb"
)

print("Loading Moondream...")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True
)

img_paths = sorted(image_dir.glob("*.png"))

for frame_idx, img_path in enumerate(img_paths):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    cv2.imshow("RGB", img)

    if frame_idx % 30 == 0:

        image = Image.open(img_path).convert("RGB")

        answer = model.query(
            image,
            """
Describe this image hierarchically.

Include:
- Environment
- Regions
- Terrain
- Objects
- Structures
- Landmarks

Return a concise tree structure.
"""
        )["answer"]

        print("\n" + "=" * 80)
        print(f"Frame {frame_idx}")
        print(img_path.name)
        print(answer)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        cv2.waitKey(0)

cv2.destroyAllWindows()