import cv2
from pathlib import Path
from ultralytics import YOLOWorld

# Load model
model = YOLOWorld("yolov8x-worldv2.pt")

# Folder containing images
# image_dir = Path(r"C:\Windows\System32\forest_frames\zed2i_left_images")

image_dir = Path(r"D:\kab3_data\rgb")

# image_dir = Path(r"D:\rgbd_dataset_freiburg2_pioneer_slam\rgbd_dataset_freiburg2_pioneer_slam\rgb")

# No custom classes -> uses built-in vocabulary
for img_path in sorted(image_dir.glob("*")):

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    results = model.predict(
        img,
        conf=0.03,
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