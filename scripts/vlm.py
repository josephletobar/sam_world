import ast
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import ollama


PROMPT = """
Return ONLY a valid Python list of concise semantic segmentation labels.

Rules:
- Maximum 8 labels
- Lowercase only
- No explanation
- No numbering
- No sentences
- If a label is already present in {vocabulary}, or is a close synonym / near-duplicate of an existing label, DO NOT repeat it

- Focus on clearly identifiable objects, structures, terrain, hazards, and human-related entities useful for search and rescue robotics
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

    def generate(self, prompt, image):

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


        return response.output_text


class OllamaClient:
    def __init__(self, model="qwen2.5vl:3b"):
        self.model = model

    def generate(self, prompt, image):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image]
                }
            ]
        )

        return response["message"]["content"]


def vlm(base64_image: str, vocabulary, client=OpenAIClient()):

    prompt = PROMPT.format(
        vocabulary=list(vocabulary)
    )

    response = client.generate(
        prompt=prompt,
        image=base64_image
    )


    sam_labels = ast.literal_eval(
        response
    )



    return sam_labels
