from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
from openai import OpenAI
import base64
from dotenv import load_dotenv
from scripts.llm import llm
from scripts.frame_dif import frame_dif

load_dotenv()

# LLM Setup
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# SAM3 Setup
overrides = dict(
    conf=0.8,
    task="segment",
    mode="predict",
    model="sam3.pt",
)
sam_predictor = SAM3SemanticPredictor(overrides=overrides)

# Video Setup
cap = cv2.VideoCapture("assets/challenge_video.mp4")

frame_count = 0

SAM_STEP = 5
CHANGE_STEP = 5

DEFAULT_LABELS = ["road", "car", "tree"]

run_gpt = False
prev_frame = None
sam_labels = []

while True:
    ret, frame = cap.read()

    if not ret: break

    frame_count += 1

    # Only run SAM every SAM_STEP frames
    if frame_count % SAM_STEP != 0: 
        continue

    # Stepped Frame Differencing Logic
    if prev_frame is None:
        prev_frame = frame
        run_gpt = True
        score = None
    elif frame_count % CHANGE_STEP == 0:
        run_gpt = frame_dif(prev_frame, frame)

    # Prepare image for LLM
    frame  = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    
    # Run LLM if significant change detected or first frame
    if run_gpt:
        print("--- LLM RUNNING ---")
        sam_labels = llm(client, base64_image, DEFAULT_LABELS)
        print(sam_labels)

    full_labels = DEFAULT_LABELS + sam_labels # Combine default and LLM labels for SAM3

    # Run SAM3 with the combined labels
    results = sam_predictor(
        frame,
        text=full_labels,
        # imgsz=448,
        save=False,
        verbose=False
    )
    annotated = results[0].plot() # Annotate the frame with SAM3 results

    # Display the annotated frame
    cv2.imshow("SAM3 Video", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()