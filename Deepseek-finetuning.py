
!pip uninstall -y transformers peft accelerate bitsandbytes trl
!pip install transformers==4.41.2 peft accelerate bitsandbytes
!pip install peft==0.8.2
!pip install accelerate==0.27.2
!pip install bitsandbytes
!pip install trl==0.7.10
!pip install datasets

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
login()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

from datasets import load_dataset, DatasetDict

dataset = DatasetDict({
    "train": load_dataset("json", data_files="train1.jsonl")["train"],
    "validation": load_dataset("json", data_files="val1.jsonl")["train"],
    "test": load_dataset("json", data_files="test1_infer.jsonl")["train"],
})

def formatting_prompts_train(example):
    messages = example["messages"]
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

    return {
        "note_id": example["note_id"],
        "prompt": user_msg,
        "response": assistant_msg
    }

def formatting_prompts_test(example):
    messages = example["messages"]
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    return {
        "note_id": example["note_id"],
        "prompt": user_msg
    }

dataset["train"] = dataset["train"].map(formatting_prompts_train, remove_columns=dataset["train"].column_names)
dataset["validation"] = dataset["validation"].map(formatting_prompts_train, remove_columns=dataset["validation"].column_names)
dataset["test"] = dataset["test"].map(formatting_prompts_test, remove_columns=dataset["test"].column_names)

!pip install trl==0.8.1

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
training_args = TrainingArguments(
    output_dir="./deepseek_lora_finetune",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    bf16=True,
    fp16=False,
    report_to="none"
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    dataset_text_field="prompt",
    max_seq_length=1024
)

train_losses = []
val_losses = []
steps = []
from transformers import TrainerCallback
class LossRecorderCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                train_losses.append(logs["loss"])
                steps.append(state.global_step)
            if "eval_loss" in logs:
                val_losses.append(logs["eval_loss"])
trainer.add_callback(LossRecorderCallback)
trainer.train()
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="Training Loss")
plt.plot(steps[:len(val_losses)], val_losses, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss during DeepSeek LoRA Fine-tuning")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

output_dir = "./deepseek_lora_finetune_adapter"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA adapter have saved to：{output_dir}")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
lora_model = PeftModel.from_pretrained(base_model, "./deepseek_lora_finetune_adapter")
lora_model.eval()

import json
from tqdm import tqdm
output_path = "deepseek_test_outputs.jsonl"
error_path = "deepseek_test_errors.jsonl"
with open(output_path, "w", encoding="utf-8") as fout, open(error_path, "w", encoding="utf-8") as ferr:
    for example in tqdm(test_dataset):
        try:
            prompt = example["prompt"]
            note_id = example["note_id"]
            input_text = (
                "<|User|> You are a medical radiology report analysis expert who is good at reading reports and extracting structured content. Please output the following structure in JSON format\n"
                “{\”Examination type\”: \"\", \”Examination aera\”: [\"\"], \”Interventional procedure\”: [\"\"]}。\n"
                f”Radiology Report：{prompt.strip()}\n\n<|Assistant|>"
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(lora_model.device)
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            fout.write(json.dumps({
                "note_id": note_id,
                "prompt": prompt,
                "reply": reply
            }, ensure_ascii=False) + "\n")

        except Exception as e:
            ferr.write(json.dumps({
                "note_id": example.get("note_id", "unknown"),
                "prompt": prompt,
                "error": str(e)
            }, ensure_ascii=False) + "\n")
print("Output saved to：", output_path)

