import ast
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import ollama
import time
import cv2
import base64
import spacy
import json


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
Return ONLY valid JSON.

{{
    "scene": "...",
    "landmarks": [...]
}}

scene:
- maximum 2 words
- describe the overall environment

landmarks:
- physical objects only
- include only objects a human operator would realistically use as a reference later
- name objects the way a human operator would naturally refer to them
- keep names short and practical
- include color or type only when it helps identify the object
- do not include background, terrain, vegetation, roads, pavement, markings, textures, walls, floors, ceilings, or ordinary natural features
- return [] if no useful reference objects are visible

Only include objects directly visible in the image.

Return ONLY valid JSON.
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

        print(response.output_text)

        response_dict = json.loads(response.output_text)

        print(response_dict)

        sam_labels = response_dict.get(
            "landmarks",
            []
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

class SceneUnderstanding:

    def __init__(self, client):

        if client == "openai":
            self.client = OpenAIClient()
        elif client == "ollama":
            self.client = OllamaClient()
        else:
            raise ValueError("Invalid client specified. Use 'openai' or 'ollama'.") 

    def get_labels(self, frame, vocabulary):
        # Prepare image for VLM
        downsized_frame = cv2.resize(frame, (960, 540))
        _, buffer = cv2.imencode(".jpg", downsized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        labels = self.client.generate(
            vocabulary=vocabulary,
            image=base64_image,
        )

        return labels
    





