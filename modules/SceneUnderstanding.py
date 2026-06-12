import ast
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import ollama
import time
import cv2
import base64
import spacy
import json
import re


nlp = spacy.load("en_core_web_sm")


# LLM_PROMPT = """
# Vocabulary:
# {vocabulary}

# Candidate labels:
# {vlm_out}

# Return a Python list containing all candidate labels that are not already in the vocabulary.

# Rules:
# - lowercase
# - remove duplicates
# - merge synonyms
# - return only a Python list

# Return ONLY:
# ["label1","label2"]
# """

OPEN_AI_VLM_PROMPT = """
Return ONLY valid JSON.

{{
"scene": "...",
"landmarks": [...]
}}

scene:

* overall environment
* maximum 2 words
* if it already exists as a synonym in {scene_vocab}, reuse the name

landmarks:

* list visible objects that could be useful landmarks
* use short object names
* prefer distinctive objects
* do not include terrain, vegetation, surfaces, markings, or clutter
* do not describe objects
* do not explain
* only include objects that are directly visible

Return ONLY valid JSON.
"""


REASONING_PROMPT = """
Vocabulary:
{vocabulary}

Candidate landmarks:
{candidate_landmarks}

You are selecting landmarks for open-world search and rescue navigation.

Most visible objects are NOT landmarks.

For each candidate landmark, ask:

Would a human realistically use this object to identify or communicate a specific location?

Keep only objects that a human would naturally reference when giving directions or describing a location.

A landmark should help another person find a unique place.

Reject objects that are generic, repetitive, continuous, temporary, or unhelpful for navigation.

Never keep:

* roads
* sidewalks
* curbs
* pavement
* ground
* dirt
* grass
* leaves
* rocks
* trees
* bushes
* vegetation
* sky
* clouds
* terrain

Vocabulary rules:

* Merge duplicates.
* Merge synonyms.
* Merge overly specific variants into a canonical landmark type.
* Prefer the most generic landmark type that preserves the object's identity.
* "yellow tripod", "metal tripod", and "camera tripod" should become "tripod".
* "wooden bench" and "metal bench" should become "bench".
* Use existing vocabulary whenever reasonable.
* Do not assume the vocabulary is correct.
* If a vocabulary term is unnecessarily specific, simplify it.
* Prefer simple landmark names.
* Use lowercase.
* One or two words maximum.

Return only a valid Python list of strings.

Do not return JSON.
Do not return explanations.
Do not return reasoning.
Do not return markdown.
Do not return any text before or after the list.
"""





def normalize_label(label):
    label = str(label).lower().strip()
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"[^a-z0-9 ]+", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def normalize_labels(labels):
    normalized = []
    seen = set()

    if not isinstance(labels, list):
        return normalized

    for label in labels:
        label = normalize_label(label)
        if not label or label in seen:
            continue

        seen.add(label)
        normalized.append(label)

    return normalized


def parse_python_list(text):
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []

    return normalize_labels(value)


def parse_scene_response(text):
    try:
        response_dict = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "", []
        response_dict = json.loads(text[start:end + 1])

    scene = normalize_label(response_dict.get("scene", ""))
    landmarks = normalize_labels(response_dict.get("landmarks", []))
    return scene, landmarks




class OpenAIClient:
    def __init__(self, model="gpt-4.1"):
        load_dotenv(override=True)

        self.client = OpenAI()
        self.model = model


    def generate(self, vocabulary, scene_vocab, image):
        vocabulary = sorted(vocabulary or [])
        scene_vocab = sorted(scene_vocab or [])

        prompt = OPEN_AI_VLM_PROMPT.format(
            scene_vocab=json.dumps(scene_vocab)
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

        scene, raw_landmarks = parse_scene_response(response.output_text)

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=REASONING_PROMPT.format(
                vocabulary=json.dumps(vocabulary),
                candidate_landmarks=json.dumps(raw_landmarks)
            )
        )

        # response = ollama.chat(
        #     model="qwen3:4b",
        #     options={
        #         "temperature": 0,
        #         "num_predict": 100
        #     },
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"{REASONING_PROMPT}\n\nLandmarks:\n{raw_landmarks}"
        #         }
        #     ]
        # )

        refined_landmarks = parse_python_list(response.output_text)



        print(refined_landmarks)

        return refined_landmarks, scene


class OllamaClient:
    def __init__(self, model="qwen2.5vl:3b"):
        self.model = model

    def generate(self, vocabulary, scene_vocab=None, image=None):

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

        labels = normalize_labels([
            chunk.text
            for chunk in doc.noun_chunks
        ])

        print(labels)

        return labels, ""

class SceneUnderstanding:

    def __init__(self, client, slam_frame=None, vocab=None, scene_vocab=None):
        self.slam_frame = slam_frame
        self.vocabulary = vocab if vocab is not None else set()
        self.scene_vocab = scene_vocab if scene_vocab is not None else set()

        if client == "openai":
            self.client = OpenAIClient()
        elif client == "ollama":
            self.client = OllamaClient()
        else:
            raise ValueError("Invalid client specified. Use 'openai' or 'ollama'.") 

    def get_labels(self):
        
        frame = self.slam_frame.rgb

        if frame is None:
            raise RuntimeError("SceneUnderstanding needs a current rgb frame")

        # Prepare image for VLM
        downsized_frame = cv2.resize(frame, (960, 540))
        _, buffer = cv2.imencode(".jpg", downsized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        labels, scene = self.client.generate(
            vocabulary=self.vocabulary,
            scene_vocab=self.scene_vocab,
            image=base64_image,
        )

        return labels, scene
    





