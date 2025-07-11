
!pip install transformers==4.41.2
!pip install peft==0.10.0
!pip install accelerate==0.29.2
!pip install trl==0.8.1
!pip install bitsandbytes
!pip install datasets

from huggingface_hub import login
login()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False 
model.eval()

from datasets import load_dataset, DatasetDict

dataset = DatasetDict({
    "train": load_dataset("json", data_files="train1.jsonl")["train"],
    "validation": load_dataset("json", data_files="val1.jsonl")["train"],
    "test": load_dataset("json", data_files="test1_infer.jsonl")["train"],
})
def formatting_prompts_train(example):
    messages = example["messages"]
    user_message = None
    assistant_message = None
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
        elif message["role"] == "assistant":
            assistant_message = message["content"]

    if user_message is None or assistant_message is None:
        raise ValueError("Missing user or assistant message!")

    return {
        "note_id": example["note_id"],
        "prompt": user_message,
        "response": assistant_message
    }
def formatting_prompts_test(example):
    messages = example["messages"]
    user_message = None
    for message in messages:
        if message["role"] == "user":
            user_message = message["content"]
    if user_message is None:
        raise ValueError("Missing user message!")
    return {
        "note_id": example["note_id"],
        "prompt": user_message
    }
dataset["train"] = dataset["train"].map(formatting_prompts_train, remove_columns=dataset["train"].column_names)
dataset["validation"] = dataset["validation"].map(formatting_prompts_train, remove_columns=dataset["validation"].column_names)
dataset["test"] = dataset["test"].map(formatting_prompts_test, remove_columns=dataset["test"].column_names)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./mistral_lora_finetune",
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

from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    dataset_text_field="prompt",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
plt.figure(figsize=(10,6))
plt.plot(steps, train_losses, label="Training Loss")
plt.plot(steps[:len(val_losses)], val_losses, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss during Fine-tuning")
plt.legend()
plt.grid()
plt.show()

output_dir = "./mistral_lora_finetune_adapter"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"have saved to the {output_dir} directory")

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
lora_model = PeftModel.from_pretrained(base_model, "./mistral_lora_finetune_adapter")
lora_model.eval()
print("The fine-tuned LoRA model is loaded！")

import json
from tqdm import tqdm
output_path = "mistral_test_outputs.jsonl"
error_path = "mistral_test_errors.jsonl"
with open(output_path, "w", encoding="utf-8") as fout, open(error_path, "w", encoding="utf-8") as ferr:
    for example in tqdm(dataset["test"]):
        try:
            prompt = example["prompt"]
            note_id = example["note_id"]

            input_text = f"<s>[INST] Please read following radiology report and identify the following information. Please output the result in JSON format as follows：” \
             “{\”examination type\”: \"\", \”examination area\”: [\"\"], \”interventional procedure\”: [\"\"]}。" \
             f"\nReports：{prompt.strip()} [/INST]"

            inputs = tokenizer(input_text, return_tensors="pt").to(lora_model.device)

            outputs = lora_model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
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
print("Test set completed! Output file:", output_path)

