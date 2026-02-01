import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Paths
model_path = r"e:\LLM Tuning\finetune_project\model_output"

def main():
    print(f"Loading fine-tuned model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nModel loaded successfully! Ready for inference.")
    print("-" * 50)
    print("Type 'exit' or 'quit' to stop.\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    while True:
        try:
            user_input = input("User (Instruction): ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Optional: Ask for separate 'Input' context if needed, but for simplicity let's assume instruction only or combine
            # If the user wants to mimic the training format:
            # Instruction: ...
            # Input: ... (Optional)
            
            prompt = f"User: {user_input}\n\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            print("\nAssistant (Thinking...):")
            _ = model.generate(
                **inputs, 
                streamer=streamer, 
                max_new_tokens=1024, 
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1
            )
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"inference error: {e}")

if __name__ == "__main__":
    main()
