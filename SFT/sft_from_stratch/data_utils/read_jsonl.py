import json


def read_jsonl(file_path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data.jsonl"
    records = read_jsonl(path)
    print(f"Loaded {len(records)} records")
    if records:
        print("First record:", json.dumps(records[0], ensure_ascii=False, indent=2))