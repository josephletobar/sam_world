import cv2
from pathlib import Path
from ultralytics import YOLOWorld

# Load model
model = YOLOWorld("yolov8s-world.pt")

# Folder containing images
# image_dir = Path(r"C:\Windows\System32\forest_frames\zed2i_left_images")

image_dir = Path(r"C:\Users\jleto\Downloads\rgbd_dataset_freiburg1_xyz\rgbd_dataset_freiburg1_xyz\rgb")

# No custom classes -> uses built-in vocabulary
for img_path in sorted(image_dir.glob("*")):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    results = model.predict(
        img,
        conf=0.25,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("YOLO-World", annotated)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):  # pause on space
        cv2.waitKey(0)

cv2.destroyAllWindows()