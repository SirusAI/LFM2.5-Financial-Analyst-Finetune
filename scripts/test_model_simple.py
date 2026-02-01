import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Paths
model_path = r"e:\LLM Tuning\finetune_project\model_output"

def main():
    print(f"Loading fine-tuned model from {model_path}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return

    print("Model loaded successfully!", flush=True)
    
    prompt_text = "Analyze the current trends in Chinese A-shares."
    if len(sys.argv) > 1:
        prompt_text = sys.argv[1]
        
    formatted_prompt = f"User: {prompt_text}\n\nAssistant:"
    
    print(f"\nGenerating response for: {prompt_text}", flush=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Model Response ---\n")
    print(response)
    print("\n----------------------\n")

if __name__ == "__main__":
    main()
