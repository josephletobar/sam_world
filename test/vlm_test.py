import time
from pathlib import Path
import ollama

MODEL = "qwen2.5vl:3b"
# MODEL = "minicpm-v"

PROMPT = "What is in this image? One sentence."

image_dir = Path(r"D:\kab3_data\rgb")

latencies = []

for img_path in sorted(image_dir.glob("*")):

    t0 = time.time()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [str(img_path)]
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 5
        }
    )

    dt = time.time() - t0
    latencies.append(dt)

    print("\n========================")
    print(img_path.name)
    print("Model:", MODEL)
    print("Latency:", round(dt, 2), "sec")
    print("Average:", round(sum(latencies) / len(latencies), 2), "sec")
    print(response["message"]["content"])

    if len(latencies) >= 20:
        break