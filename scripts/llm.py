import ast

def llm(client, base64_image: str, DEFAULT_LABELS):

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
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
                            - If a label is already present in {DEFAULT_LABELS}, or is a close synonym / near-duplicate of an existing label, DO NOT repeat it

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
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    sam_labels = ast.literal_eval(
        response.output_text
    )

    return sam_labels