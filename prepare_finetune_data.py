import json

input_file = "feedback_log.jsonl"
output_file = "finetune_data.jsonl"

with open(input_file, "r") as f:
    lines = f.readlines()

training_data = []

for line in lines:
    try:
        entry = json.loads(line)
        if entry.get("rating") == 1:
            prompt = entry.get("query", "").strip()
            completion = entry.get("answer", "").strip()
            if prompt and completion:
                training_data.append({
                    "prompt": prompt,
                    "completion": completion
                })
    except json.JSONDecodeError:
        continue

with open(output_file, "w") as out:
    for item in training_data:
        out.write(json.dumps(item) + "\n")

print(f"✅ Saved {len(training_data)} prompt-completion pairs to {output_file}")
