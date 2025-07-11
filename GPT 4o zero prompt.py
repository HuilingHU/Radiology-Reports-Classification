from openai import OpenAI
import json
import time
from tqdm import tqdm

client = OpenAI(api_key="XXX")

input_path = "radiology_prompts.jsonl"
output_path = "gpt4o_outputs.jsonl"
log_path = "gpt4o_errors.jsonl"

def safe_parse_json(text):
    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_path, "w", encoding="utf-8") as fout, \
     open(log_path, "w", encoding="utf-8") as ferr:

    for line in tqdm(lines):
        obj = json.loads(line)
        note_id = obj["note_id"]
        prompt = obj["prompt"]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical radiology report analysis expert who is good at understanding imaging reports and extracting structured content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            reply = response.choices[0].message.content.strip()

            parsed_json, error = safe_parse_json(reply)

            result = {
                "note_id": note_id,
                "prompt": prompt,
                "gpt4o_output": reply,
                "parsed": parsed_json if parsed_json else {},
                "error": error
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            ferr.write(json.dumps({
                "note_id": note_id,
                "error": str(e)
            }, ensure_ascii=False) + "\n")

        time.sleep(1.2)
