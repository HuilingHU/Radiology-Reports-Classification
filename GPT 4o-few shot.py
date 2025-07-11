from openai import OpenAI
import json
import time
from tqdm import tqdm

client = OpenAI(api_key="XXX") 

input_path = "radiology_prompts.jsonl"
output_path = "gpt4o_fewshot_outputs.jsonl"
log_path = "gpt4o_fewshot_errors.jsonl"


fewshot_intro = """请参考以下示例格式，分析新的放射学报告内容：

示例1：
报告文本：
HISTORY:  An ___ female with non-small cell lung cancer and possible pneumonia.
...
输出JSON：
{
  "检查类型": "CT",
  "检查部位": ["胸部"],
  "介入操作": ["-"]
}

示例2：
报告文本：
INDICATION:  ___ year old man with cardiogenic pulmonary edema...
...
输出JSON：
{
  "检查类型": "X线",
  "检查部位": ["胸部"],
  "介入操作": ["-"]
}

示例3：
报告文本：
EXAMINATION: CHEST (PORTABLE AP)
...
输出JSON：
{
  "检查类型": "X线",
  "检查部位": ["胸部"],
  "介入操作": ["-"]
}

请分析以下报告文本，并输出相同结构的JSON结果：
"""


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
        raw_prompt = obj["prompt"]

    
        full_prompt = fewshot_intro + raw_prompt

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical radiology report analysis expert who is good at understanding imaging reports and extracting structured content."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2
            )
            reply = response.choices[0].message.content.strip()

        
            if reply.startswith("```json"):
                reply = reply[len("```json"):].strip()
            if reply.endswith("```"):
                reply = reply[:-3].strip()

            parsed_json, error = safe_parse_json(reply)

    
            result = {
                "note_id": note_id,
                "parsed": parsed_json if parsed_json else {},
            }
            if error:
                result["error"] = error

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            ferr.write(json.dumps({
                "note_id": note_id,
                "error": str(e)
            }, ensure_ascii=False) + "\n")

        time.sleep(1.2)
