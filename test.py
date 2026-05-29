from ultralytics.models.sam import SAM3SemanticPredictor
import cv2

from openai import OpenAI
client = OpenAI()

import base64
from dotenv import load_dotenv

load_dotenv()

print("hi")

overrides = dict(
    conf=0.8,
    task="segment",
    mode="predict",
    model="sam3.pt",
)
predictor = SAM3SemanticPredictor(overrides=overrides)

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

    if frame_count % SAM_STEP != 0: 
        continue

    # stepped frame similarity 
    if prev_frame is None:
        prev_frame = frame
        run_gpt = True
        score = None
    elif frame_count % CHANGE_STEP == 0:
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_curr, gray_prev)
        score = diff.mean()

        print(f"DIFFERENCE SCORE: {score}")

        if score > 50:
            run_gpt = True
        else:
            run_gpt = False

        prev_frame = frame
    else:
        run_gpt = False


    frame  = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    
    if run_gpt:

        print("GPT CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
                            Return ONLY a valid Python list of concise semantic segmentation labels.

                            Rules:
                            - Maximum 8 labels
                            - Lowercase only
                            - No explanation
                            - No numbering
                            - No sentences
                            - Focus on visible robotics-relevant entities and terrain
                            - If a label is already present in {DEFAULT_LABELS}, DO NOT repeat it.

                            Example:
                            ["sidewalk", "building", "sign", "person"]
                        """,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }],
        )

        sam_labels = eval(response.output_text)

        print(sam_labels)

    full_labels = DEFAULT_LABELS + sam_labels

    results = predictor(
        frame,
        text=full_labels,
        # imgsz=448,
        save=False,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("SAM3 Video", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()