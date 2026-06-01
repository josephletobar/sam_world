from ultralytics.engine.results import Results
import numpy as np
import cv2
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed(frame : np.ndarray, mask: torch.Tensor):

    mask = mask.cpu().numpy()
    segmented = frame.copy()
    segmented[mask == 0] = 0
    ys, xs = np.where(mask > 0.5)
    cropped = segmented[
        ys.min():ys.max(),
        xs.min():xs.max()
    ]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(cropped)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features = image_features.cpu().numpy().squeeze().tolist()
    return image_features