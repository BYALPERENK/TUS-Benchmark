OpenRouter LLM Model Evaluation Framework
=========================================

A comprehensive Python framework for evaluating multiple AI models through OpenRouter API. This project enables automated testing of various language models on custom question sets with different prompting strategies, followed by automated evaluation and analysis.

### Features

- Multi-Model Testing: Test multiple AI models simultaneously via OpenRouter API
- Concurrent Processing: Efficient async processing with configurable concurrency limits
- Flexible Prompting: Support for multiple prompt strategies per question set
- Automated Evaluation: AI-powered evaluation of model responses
- Rate Limiting: Built-in rate limiting to respect API constraints

### Project Structure

├── openrouter_processor.py   # Core processing logic for model evaluation  
├── make_test.py              # Main script to run model tests  
├── eval_answer.py            # Automated evaluation of model responses  
├── requirements.txt          # Python dependencies  
├── model_info/       
│   └── ask_models.py         # Lists the models and their features that we can access with the OpenRouter API   
├── keys/  
│   └── openrouterai_api_key.txt  # Your OpenRouter API key (required)  
├── questions_data/  
│   ├── questions.json       # Question dataset  
│   └── prompts.json         # Prompt templates  
├── results/                 # Generated results (auto-created)  
└── evaluation/              # Evaluation results (auto-created)  

### Requirements  
  
Python: 3.9 or higher  
OpenRouter API Key: Required for accessing AI models  
Dependencies: Listed in requirements.txt

### Usage

1) Create and add your API key

Get an OpenRouter API key from your account and save it to:
    keys/openrouterai_api_key.txt

The file should contain only the key, for example:
    sk-or-xxxxxxxxxxxxxxxx

------------------------------------------------------
2) Inspect available models (optional but recommended)

Use the helper to fetch the latest model list plus pricing/context details via OpenRouter:
    python model_info/ask_models.py

This generates the following files in model_info/:
    - models_pretty.txt  -> pretty JSON dump of the full response
    - models_simple.txt  -> human-readable list
    - models_table.txt   -> quick table view (name / id / context / pricing)

Use these to decide which models you want to test.

------------------------------------------
3) Prepare questions.json and prompts.json

Place your datasets here:
    questions_data/questions.json
    questions_data/prompts.json

The repo already includes example JSONs—follow their format.
- questions.json: list of question objects (e.g., question_no, question_eng, etc.)
- prompts.json: a dict with "prompts" mapping prompt keys to objects containing a name/template (align with how your code reads it)

Tip: Keep the dataset small at first to validate the pipeline, then scale up.

-------------------
4) Configure models

Edit make_test.py and set the models you want to benchmark, for example:
```
    models = [
        "ai21/jamba-mini-1.7",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        # Add more models as needed
    ]
```
You can also tune:
    concurrent_requests    (per model run)
    max_retries / retry_delay

These values are defined in make_test.py and passed into process_questions_with_openrouter.

Note: Respect each provider’s rate limits. If you see 429 errors, reduce concurrency or increase retry_delay.

-------------------------------------
5) Run model tests (generation phase)

This step sends your questions + prompts to each selected model and saves raw outputs.
    python make_test.py

Outputs:
    results/<sanitized_model_name>/*.json
        - responses per prompt with token usage
    results/<sanitized_model_name>/<sanitized_model_name>_stats.txt
        - total tokens per prompt

File names include timestamps, for example:
    results/openai_gpt_4o_mini/openai_gpt_4o_mini_Baseline_20250101_123045.json

----------------------------------------------
6) Run automated evaluation (evaluation phase)

This step reads the JSONs from results/** and produces analyzed files with extracted answers and notes.
    python eval_answer.py

What it does:
- Uses openai/gpt-4o-mini by default as the analyzer (configurable in eval_answer.py)
- Applies a rate limiter and concurrency control to avoid throttling
- Parses each LLM output into a JSON object like:
      { "llm_answer": "A|B|C|D|E|NOT CHOOSEN|Failed",
        "evaluation_note": "string (optional)" }

Outputs:
    evaluation/<source_model_name>/*_analyzed_<timestamp>.json
        - evaluated results
    evaluation/<source_model_name>/eval_<source_model_name>_by_<analyzer>_token_stats.txt
        - token totals for evaluation

Quick smoke test tip:
- Set try_mode=True in process_model/process_json_file to limit items for a faster run.


---------------------------------------------------------------------
ARTICLE
---------------------------------------------------------------------
A detailed article is available here:
    [Article link – soon]

## License

This project is licensed under the MIT License.

