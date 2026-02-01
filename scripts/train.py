import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# Configuration
max_seq_length = 2048
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_in_4bit = False 

# Paths
model_path = r"e:\LLM Tuning\LLM models\LFM2.5-1.2B-Thinking"
data_path = r"e:\LLM Tuning\finetune_project\data\merged_dataset.jsonl"
output_dir = r"e:\LLM Tuning\finetune_project\model_output"

def main():
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = dtype,
        device_map = "auto",
        trust_remote_code = True,
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files = data_path, split = "train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            if input:
                prompt = f"User: {instruction}\n{input}\n\nAssistant: {output}"
            else:
                prompt = f"User: {instruction}\n\nAssistant: {output}"
            texts.append(prompt + tokenizer.eos_token)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("Starting training with HF Trainer...")
    
    # SFTConfig setup - Only TrainingArguments params + dataset_text_field
    training_args = SFTConfig(
        output_dir = "outputs",
        dataset_text_field = "text",
        max_length = max_seq_length,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = (dtype == torch.float16),
        bf16 = (dtype == torch.bfloat16),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = "no",
        dataset_num_proc = 2,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
        peft_config = peft_config,
    )

    trainer.train()

    print("Training complete. Saving merged model...")
    # Merge LoRA and Save
    model = model.merge_and_unload() 
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
