from ultralytics import YOLOWorld

class PriorityObjectDetector:
    def __init__(self):
        self.model = YOLOWorld("yolov8x-worldv2.pt")
        self.model.set_classes([
            "person",
            "vehicle",
            "backpack",
            "dog"
        ])
        self.prev_yolo_labels = None
        self.detected_objects = None
        self.yolo_boxes = None

    def _yolo_diff(self, frame):
        
        yolo_objs = self.model.predict(
            frame,
            conf=0.8,
            verbose=False
        )
        labels = {
            yolo_objs[0].names[int(box.cls)]
            for box in yolo_objs[0].boxes
        }

        self.detected_objects = labels

        self.yolo_boxes = yolo_objs[0].boxes

        yolo_feature = 1.0 if self.prev_yolo_labels != labels  else 0.0
        if yolo_feature == 1.0: 
            self.prev_yolo_labels = labels
            print("YOLO DIFF DETECTED")
            return True
        else: 
            self.prev_yolo_labels = labels
            return False

    def detect(self, frame):
    
        if self._yolo_diff(frame):
            return list(self.detected_objects)

        return []