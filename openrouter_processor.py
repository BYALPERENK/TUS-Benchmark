import json
import asyncio
from datetime import datetime
import os
from typing import Dict, List
from openai import AsyncOpenAI

def sanitize_model_name(model_name: str) -> str:
    """Makes the model name suitable for the file system"""
    # Replace characters that are not available on Windows
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '-', '.']
    safe_name = model_name
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    return safe_name

async def process_questions_with_openrouter(
    model_name: str,
    questions: List[Dict],
    prompts: Dict,
    api_key: str,
    concurrent_requests: int = 5,
    max_retries: int = 3,
    retry_delay: int = 5,
    base_folder: str = None
) -> None:
    
    safe_model_name = sanitize_model_name(model_name)
    
    # Create a folder to save results
    results_folder = os.path.join(base_folder, safe_model_name)
    os.makedirs(results_folder, exist_ok=True)

    # Dictionary to hold statistics
    stats = {}
    
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    async def process_single_question(question: Dict, prompt: str) -> Dict:
        """Processes a single question"""
        print(f"Processing question {question['question_no']} for {model_name} with {prompt['name']} prompt")
        
        full_prompt = f"{prompt}\n\n{question['question_eng']}"
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    extra_headers={
                        "X-Title": "DrAI"
                    },
                    # extra_body={
                    #     "provider": {
                    #         "quantizations": ["fp16", "bf16"]
                    #         # "quantizations": ["fp16", "bf16", "fp8"]
                    #     }
                    # }
                )
                
                # Response control
                if not response or not response.choices or not response.choices[0].message:
                    raise ValueError("Invalid API response format")
                
                result = question.copy()
                result["llm_solution"] = response.choices[0].message.content
                
                # Usage control
                if hasattr(response, 'usage') and response.usage:
                    result["token_count"] = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "response_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0)
                    }
                else:
                    result["token_count"] = {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "total_tokens": 0
                    }
                
                return result
                
            except Exception as e:
                error_msg = f"Error processing question {question['question_no']} (Attempt {attempt + 1}/{max_retries}): {str(e)}"
                print(error_msg)
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    return {
                        **question,
                        "llm_solution": f"Error after {max_retries} attempts: {str(e)}",
                        "token_count": {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
                    }

    async def process_prompt(prompt_key: str, prompt_data: Dict):
        """Processes all questions for a prompt"""
        print(f"\nProcessing {prompt_data['name']} prompt for {model_name}...")
        
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def process_with_semaphore(question):
            async with semaphore:
                return await process_single_question(question, prompt_data)
        
        tasks = [process_with_semaphore(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        total_prompt_tokens = sum(r["token_count"]["prompt_tokens"] for r in results)
        total_response_tokens = sum(r["token_count"]["response_tokens"] for r in results)
        total_tokens = sum(r["token_count"]["total_tokens"] for r in results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{safe_model_name}_{prompt_data['name']}_{timestamp}.json"
        file_path = os.path.join(results_folder, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        stats[prompt_key] = {
            'prompt_tokens': total_prompt_tokens,
            'response_tokens': total_response_tokens,
            'total_tokens': total_tokens
        }
        
        print(f"Completed {prompt_data['name']} prompt for {model_name}")
        print(f"Results saved to '{file_path}'")
    
    # Process all prompts
    for prompt_key, prompt_data in prompts['prompts'].items():
        await process_prompt(prompt_key, prompt_data)

    # Save statistics
    stats_file = os.path.join(results_folder, f"{safe_model_name}_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"{model_name}\n\n")
        for prompt_key, prompt_data in prompts['prompts'].items():
            prompt_stats = stats[prompt_key]
            f.write(f"{prompt_data['name'].title()} Prompt\n")
            f.write("Final token statistics:\n")
            f.write(f"Total prompt tokens: {prompt_stats['prompt_tokens']}\n")
            f.write(f"Total response tokens: {prompt_stats['response_tokens']}\n")
            f.write(f"Total combined tokens: {prompt_stats['total_tokens']}\n\n")
    
    print(f"\nAll prompts processed for {model_name}. Statistics saved to '{stats_file}'")