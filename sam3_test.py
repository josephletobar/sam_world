import cv2
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    verbose=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

image_dir = Path(r"D:\kab3_data\rgb")

for img_path in sorted(image_dir.glob("*")):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    predictor.set_image(img)

    results = predictor(
        text=[
            "person",
            "tripod",
            "bicycle",
            "backpack",
            "car",
            "tree",
            "bench"
        ]
    )

    annotated = img.copy()

    if len(results):
        r = results[0]

        if r.masks is not None:
            annotated = r.plot()

    cv2.imshow("SAM3", annotated)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        cv2.waitKey(0)

cv2.destroyAllWindows()