import asyncio
import json
import os

from openrouter_processor import process_questions_with_openrouter

async def main():
    current_dir = os.getcwd()
    
    results_folder = os.path.join(current_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    questions_path = os.path.join(current_dir, "questions_data", "questions.json")
    prompts_path = os.path.join(current_dir, "questions_data", "prompts.json")
    api_key_path = os.path.join(current_dir, "keys", "openrouterai_api_key.txt")

    # Check for the existence of files
    for path in [questions_path, prompts_path, api_key_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    with open(api_key_path, 'r') as f:
        api_key = f.read().strip()
    
    # Models
    models = [
        # "ai21/jamba-large-1.7",
        "ai21/jamba-mini-1.7", # Demo models
        "openai/gpt-5-nano"
    ]

    # Process each model
    for model in models:
        print(f"\nProcessing model: {model}")
        
        await process_questions_with_openrouter(
            model_name=model,
            questions=questions,
            prompts=prompts,
            api_key=api_key,
            concurrent_requests=40,
            max_retries=5,
            retry_delay=3,
            base_folder=results_folder
        )

if __name__ == "__main__":
    asyncio.run(main())