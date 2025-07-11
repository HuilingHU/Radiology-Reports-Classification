
!pip install -U transformers accelerate peft bitsandbytes --quiet

from huggingface_hub import login
login()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

input_path = "radiology_prompts.jsonl"
output_path = "mistral_outputs.jsonl"
error_path = "mistral_errors.jsonl"
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
with open(output_path, "w", encoding="utf-8") as fout, open(error_path, "w", encoding="utf-8") as ferr:
    for line in tqdm(lines):  
        obj = json.loads(line)
        note_id = obj["note_id"]
        prompt = obj["prompt"]
        input_text = f"<s>[INST] {prompt} [/INST]"
        try:
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            fout.write(json.dumps({
                "note_id": note_id,
                "reply": reply
            }, ensure_ascii=False) + "\n")
        except Exception as e:
            ferr.write(json.dumps({
                "note_id": note_id,
                "error": str(e)
            }, ensure_ascii=False) + "\n")

