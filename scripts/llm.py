import ast

def llm(client, base64_image: str, DEFAULT_LABELS):

    response = client.chat.completions.create(
        model="qwen2.5vl:3b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
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
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    sam_labels = ast.literal_eval(
        response.choices[0].message.content
    )

    return sam_labels