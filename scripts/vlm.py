import ast
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import ollama
import time
import cv2
import base64
import spacy

nlp = spacy.load("en_core_web_sm")


LLM_PROMPT = """
Vocabulary:
{vocabulary}

Candidate labels:
{vlm_out}

Return a Python list containing all candidate labels that are not already in the vocabulary.

Rules:
- lowercase
- remove duplicates
- merge synonyms
- return only a Python list

Return ONLY:
["label1","label2"]
"""

OPEN_AI_VLM_PROMPT = """
Return ONLY a valid Python list of concise semantic segmentation labels.

Existing vocabulary:
{vocabulary}

Candidate labels from a vision model (optional):
{vlm_out}

If candidate labels are provided:
- Merge synonyms and near-duplicates
- Remove labels already present in the vocabulary
- Keep only labels relevant to search and rescue

Rules:
- Maximum 8 labels
- Lowercase only
- No explanation
- No numbering
- No sentences
- If a label is already present in the existing vocabulary, or is a close synonym / near-duplicate of an existing label, DO NOT repeat it, merge them into one if it appears. 

- Identify every distinct object visible in the scene. Be exhaustive with objects whenever they can be recognized with reasonable confidence. Output object names only.
- Prefer discrete, distinguishable entities that can be individually localized or tracked
- Avoid broad background regions or generic structural surfaces unless they are mission-relevant obstacles or landmarks
- Do NOT include generic surfaces like "wall", "floor", "ceiling", or "room" unless uniquely important to navigation or hazard assessment
- ONLY include entities directly visible in the provided image
- Prefer concrete physical entities over abstract scene descriptions
- Include obstacles, access points, debris, vehicles, infrastructure, and survivors when visible
- Avoid vague labels like "object", "area", or "environment"
- Prioritize labels that improve navigation, localization, scene understanding, or rescue decision-making

Example:
["sidewalk", "building", "sign", "person"]
"""


class OpenAIClient:
    def __init__(self, model="gpt-4.1"):
        load_dotenv(override=True)

        self.client = OpenAI()
        self.model = model


    def generate(self, vocabulary, image):

        prompt = OPEN_AI_VLM_PROMPT.format(
            vocabulary=list(vocabulary),
            vlm_out=""
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image}"
                        }
                    ]
                }
            ]
        )

        sam_labels = ast.literal_eval(
            response.output_text
        )

        return sam_labels


class OllamaClient:
    def __init__(self, model="qwen2.5vl:3b"):
        self.model = model

    def generate(self, vocabulary, image):

        t0 = time.time()
        vlm_response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": """
                        List every visible object.

                        Output only object names separated by commas.

                        Do not explain.
                        Do not describe the scene.
                        Do not repeat this instruction.
                        """,
                    "images": [image]
                }
            ]
        )

        vlm_response = vlm_response["message"]["content"]
        print(vlm_response)
        print("VLM:", time.time() - t0)


        # prompt = LLM_PROMPT.format(
        #     vocabulary=list(vocabulary),
        #     vlm_out=vlm_response
        # )
        # t0 = time.time()
        # response = ollama.chat(
        #     model="llama3.2:3b",
        #     options={
        #         "temperature": 0,
        #         "num_predict": 50
        #     },
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        # )

        # print(vocabulary)
        # print(response["message"]["content"])
        # print("LLM:", time.time() - t0)

        doc = nlp(vlm_response)

        labels = [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
        ]

        print(labels)

        return labels



def vlm(frame, vocabulary, client=OpenAIClient()):

    # Prepare image for VLM
    downsized_frame = cv2.resize(frame, (960, 540))
    _, buffer = cv2.imencode(".jpg", downsized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    base64_image = base64.b64encode(buffer).decode("utf-8")

    labels = client.generate(
        vocabulary=vocabulary,
        image=base64_image,
    )




    return labels

