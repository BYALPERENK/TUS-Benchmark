import os
import re
import json
import asyncio
import aiofiles
import time
from openai import AsyncOpenAI
from datetime import datetime
from glob import glob
from typing import Dict, Any, List
from openrouter_processor import sanitize_model_name

class RateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.rate_limit, self.tokens + time_passed * self.rate_limit)
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate_limit
                await asyncio.sleep(sleep_time)
                self.tokens = 1
                
            self.tokens -= 1
            self.last_update = now

def extract_llm_answer_and_evaluation(json_str: str) -> tuple[str, str]:
    """
    Extracts the llm_answer and evaluation_note values from the JSON string.
    Format:
    {
    "llm_answer": "A",
    "evaluation_note": "None"
    }
    """
    pattern = r'\{\s*[\n\r]*\s*"llm_answer"\s*:\s*"([^"]+)"\s*,\s*[\n\r]*\s*"evaluation_note"\s*:\s*"([^"]*)"\s*[\n\r]*\s*\}'
    match = re.search(pattern, json_str, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None

async def analyze_question(
    question: Dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    max_retries: int,
    retry_delay: int
) -> Dict[str, Any]:
    prompt_text = f"""
Below is a question and a solution for that question. Review them and write the marked option of the question in json format.

Expected output format examples:
{{
"llm_answer": "A",
"evaluation_note": "None"
}}
Another example:
{{
"llm_answer": "B",
"evaluation_note": "None"
}}
Another example:
{{
"llm_answer": "C",
"evaluation_note": "None"
}}
Another example:
{{
"llm_answer": "D",
"evaluation_note": "None"
}}
Another example:
{{
"llm_answer": "E",
"evaluation_note": "None"
}}

If more than one option is selected, no option is selected, or an answer like I don't know is given, write a json like this and write the detected error in the notes field. An example on the subject is below.
{{
"llm_answer": "NOT CHOOSEN",
"evaluation_note": "2 answers selected"
}}

If you want to give a note about the solution to the person who will evaluate the answers, you can write it in the notes section. An example on the subject is below. For example context loss, repetition error, internal consistency error etc.
{{
"llm_answer": "NOT CHOOSEN",
"evaluation_note": "There is a repetition error in the llm answer."
}}

Question:
{question.get("question_eng", "")}

Solution:
{question.get("llm_solution", "")}


Do not try to solve the question, identify the marked option and write it in the specified json format. 
"""

    async with semaphore:
        for attempt in range(max_retries):
            try:
                await rate_limiter.acquire()
                
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0
                )

                if not response.choices or not response.choices[0].message:
                    raise ValueError("Invalid API response format")

                content = response.choices[0].message.content
                usage = response.usage if hasattr(response, 'usage') else None

                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

                llm_answer, evaluation_note = extract_llm_answer_and_evaluation(content)

                if llm_answer is None:
                    llm_answer = "Failed"
                    evaluation_note = "Failed to parse JSON output"

                question["llm_answer"] = llm_answer
                question["evaluation_note"] = evaluation_note
                question["token_count_eval_mode"] = {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                return question

            except Exception as e:
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    question["llm_answer"] = "NOT EVALUATED"
                    question["evaluation_note"] = f"Error after {max_retries} attempts: {str(e)}"
                    question["token_count_eval_mode"] = {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "total_tokens": 0
                    }
                    return question

async def process_json_file(
    json_path: str,
    client: AsyncOpenAI,
    model_name: str,
    output_base: str,
    max_retries: int,
    retry_delay: int,
    concurrency: int,
    rate_limit: float,
    try_mode: bool
) -> Dict[str, int]:
    print(f"\n--- Processing JSON: {json_path} ---")

    source_model_name = os.path.basename(os.path.dirname(json_path))

    model_eval_dir = os.path.join(output_base, source_model_name)
    os.makedirs(model_eval_dir, exist_ok=True)

    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)

    if try_mode:
        data = data[:10]

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = RateLimiter(rate_limit)
    tasks = []
    
    for i, question in enumerate(data):
        print(f"  Processing question {i+1}...")
        tasks.append(
            analyze_question(
                question, client, model_name, semaphore, 
                rate_limiter, max_retries, retry_delay
            )
        )

    results = await asyncio.gather(*tasks)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_sum = 0

    for r in results:
        tk = r.get("token_count_eval_mode", {})
        total_prompt_tokens += tk.get("prompt_tokens", 0)
        total_completion_tokens += tk.get("response_tokens", 0)
        total_tokens_sum += tk.get("total_tokens", 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.basename(json_path)
    filename_without_ext = os.path.splitext(original_filename)[0]
    out_name = f"{filename_without_ext}_analyzed_{timestamp}.json"
    out_path = os.path.join(model_eval_dir, out_name)

    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=4))

    print(f"  => New JSON saved: {out_path}")

    return {
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_completion_tokens,
        "total_tokens": total_tokens_sum
    }

async def process_model(
    model_name: str,
    api_key: str,
    results_folder: str,
    try_mode: bool = False,
    concurrency: int = 100,
    max_retries: int = 3,
    retry_delay: int = 5,
    rate_limit: float = 20
) -> None:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        evaluation_dir = "evaluation"
        os.makedirs(evaluation_dir, exist_ok=True)

        stats = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }

        json_files = glob(os.path.join(results_folder, "*.json"))

        if not json_files:
            print(f"No JSON files found in {results_folder}")
            return

        if try_mode:
            json_files = json_files[:2]  # Test mode only 2 files

        for jf in json_files:
            file_stats = await process_json_file(
                json_path=jf,
                client=client,
                model_name=model_name,
                output_base=evaluation_dir,
                max_retries=max_retries,
                retry_delay=retry_delay,
                concurrency=concurrency,
                rate_limit=rate_limit,
                try_mode=try_mode
            )
            
            stats["prompt_tokens"] += file_stats["prompt_tokens"]
            stats["response_tokens"] += file_stats["response_tokens"]
            stats["total_tokens"] += file_stats["total_tokens"]

        # Save the statistics
        source_model_name = os.path.basename(results_folder)
        model_eval_dir = os.path.join(evaluation_dir, source_model_name)
        os.makedirs(model_eval_dir, exist_ok=True)

        # Include the name of the model being analyzed
        analyzer_model_name = sanitize_model_name(model_name)
        token_stat_file = os.path.join(
            model_eval_dir, 
            f"eval_{source_model_name}_by_{analyzer_model_name}_token_stats.txt"
        )

        async with aiofiles.open(token_stat_file, "w", encoding="utf-8") as f:
            await f.write(f"Source Model: {source_model_name}\n")
            await f.write(f"Analyzer Model: {model_name}\n\n")
            await f.write("Final token statistics:\n")
            await f.write(f"Total prompt tokens: {stats['prompt_tokens']}\n")
            await f.write(f"Total response tokens: {stats['response_tokens']}\n")
            await f.write(f"Total combined tokens: {stats['total_tokens']}\n")

        print(f"\nCompleted processing {source_model_name}")
        print(f"Statistics saved => {token_stat_file}")

    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    finally:
        await client.close()

async def main():

    api_key_path = os.path.join("keys", "openrouterai_api_key.txt")
    if not os.path.exists(api_key_path):
        raise FileNotFoundError(f"API key not found: {api_key_path}")
        
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    # Model to make the inference
    model = "openai/gpt-4o-mini"

    # Find all subdirectories in the results folder
    results_dir = "results"
    if not os.path.exists(results_dir):
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    subfolders = [f.path for f in os.scandir(results_dir) if f.is_dir()]
    
    if not subfolders:
        raise NotADirectoryError(f"No subdirectories found in {results_dir}")

    print(f"\nProcessing with model: {model}")
    print(f"Found {len(subfolders)} subdirectories to analyze")

    # Process for each subfolder
    for subfolder in subfolders:
        print(f"\nProcessing subfolder: {subfolder}")
        try:
            await process_model(
                model_name=model,
                api_key=api_key,
                results_folder=subfolder,
                try_mode=False,  # Test mode
                concurrency=100,
                max_retries=3,
                retry_delay=5,
                rate_limit=50.0
            )
        except Exception as e:
            print(f"Error processing {subfolder}: {e}")
            continue  

if __name__ == "__main__":
    asyncio.run(main())