import argparse
import json
from pathlib import Path


def strip_embedding_attrs(value):
    if isinstance(value, dict):
        return {
            key: strip_embedding_attrs(item)
            for key, item in value.items()
            if "embedding" not in key
        }

    if isinstance(value, list):
        return [strip_embedding_attrs(item) for item in value]

    return value


def main():
    parser = argparse.ArgumentParser(
        description="Remove embedding attributes from a NetworkX JSON graph."
    )
    parser.add_argument("input", help="Path to the input graph JSON file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path for the cleaned JSON file. Defaults to <input>_clean.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
    )

    with input_path.open("r", encoding="utf-8") as f:
        graph_data = json.load(f)

    cleaned_data = strip_embedding_attrs(graph_data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)
        f.write("\n")

    print(f"Wrote cleaned graph to {output_path}")


if __name__ == "__main__":
    main()
