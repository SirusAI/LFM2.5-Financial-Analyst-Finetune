import glob
import os
import json

def merge_datasets(input_dir, output_file):
    print(f"Searching for .jsonl files in {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    if not files:
        print("No .jsonl files found!")
        return

    print(f"Found {len(files)} files. Merging...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files:
            print(f"Processing {os.path.basename(file_path)}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():  # Skip empty lines
                            outfile.write(line)
                            total_lines += 1
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    print(f"Successfully merged {len(files)} files into {output_file}")
    print(f"Total lines: {total_lines}")

if __name__ == "__main__":
    # Define paths
    INPUT_DIR = r"e:\LLM Tuning\DataSanity\output"
    OUTPUT_FILE = r"e:\LLM Tuning\finetune_project\data\merged_dataset.jsonl"
    
    merge_datasets(INPUT_DIR, OUTPUT_FILE)
