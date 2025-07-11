!pip install -U transformers accelerate peft bitsandbytes --quiet

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
model_name = "deepseek-ai/deepseek-llm-7b-chat"
offload_path = "/content/offload_folder"
os.makedirs(offload_path, exist_ok=True)

!huggingface-cli login

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_auth_token=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
    offload_folder=offload_path,
    use_auth_token=True,
    local_files_only=False 
)
model.eval()
print("The model is loaded and the device is mapped as follows：")
print(model.hf_device_map)

import json
from tqdm import tqdm
input_path = "radiology_prompts.jsonl"
output_path = "deepseek_outputs.jsonl"
error_path = "deepseek_errors.jsonl"
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
with open(output_path, "w", encoding="utf-8") as fout, \
     open(error_path, "w", encoding="utf-8") as ferr:
    for line in tqdm(lines):
        obj = json.loads(line)
        note_id = obj["note_id"]
        prompt = obj["prompt"]
        input_text = f"""You are a medical radiology report analysis expert who is good at reading reports and extracting structured content. Please output the following structure in JSON format：
{{
  “Examination type”: “”,
  “Examination area”: [],
  “Interventional procedure”: []
}}
Radiology Report：
{prompt}
"""
        try:
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if reply.startswith("```json"):
                reply = reply[len("```json"):].strip()
            if reply.endswith("```"):
                reply = reply[:-3].strip()
            fout.write(json.dumps({
                "note_id": note_id,
                "reply": reply
            }, ensure_ascii=False) + "\n")
        except Exception as e:
            ferr.write(json.dumps({
                "note_id": note_id,
                "error": f"InferenceError: {str(e)}"
            }, ensure_ascii=False) + "\n")

