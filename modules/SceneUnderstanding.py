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

* overall environment
* maximum 2 words

landmarks:

* list visible physical objects
* prefer objects that could be used as landmarks
* each landmark must be a single identifiable object
* use short natural descriptions
* include only details needed to distinguish the object
* do not include terrain, surfaces, clutter, or regions
* include as many relevant objects as are visible

Only describe what is directly visible.

Return ONLY valid JSON.
"""

REASONING_PROMPT = """
You are given a list of candidate landmarks.

For each landmark, ask:

Would a search and rescue operator realistically use this object as a landmark when communicating with another human?

Keep objects that a search and rescue operator might realistically reference when describing a location.

Prefer distinctive and identifiable objects.

Remove only obvious clutter, surfaces, terrain, and broad scene descriptions.

Simplify descriptions into short natural landmark names.

Return only a valid Python list of strings.

Do not return JSON.
Do not return explanations.
Do not return reasoning.
Do not return markdown.
Do not return any text before or after the list.
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

        raw_landmarks = response_dict.get(
            "landmarks",
            []
        )

        response = self.client.responses.create(
            model="gpt-5",
            input=f"{REASONING_PROMPT}\n\nLandmarks:\n{raw_landmarks}"
        )

        refined_landmarks = ast.literal_eval(response.output_text)

        print(refined_landmarks)

        return refined_landmarks


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
    





