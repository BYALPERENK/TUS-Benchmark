import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import requests
import json
from tabulate import tabulate

api_key_path = os.path.join(parent_dir, "keys", "openrouterai_api_key.txt")

with open(api_key_path, 'r') as f:
        api_key = f.read().strip()

headers = {
    'Authorization': f'Bearer {api_key}'
}

response = requests.get('https://openrouter.ai/api/v1/models', headers=headers)
models = response.json()

with open(os.path.join(current_dir, 'models_pretty.txt'), 'w', encoding='utf-8') as f:
    json.dump(models, f, indent=4, ensure_ascii=False)

with open(os.path.join(current_dir, 'models_simple.txt'), 'w', encoding='utf-8') as f:
    for model in models['data']:
        f.write(f"Model Name: {model['name']}\n")
        f.write(f"ID: {model['id']}\n")
        f.write(f"Context Length: {model['context_length']}\n")
        f.write(f"Description: {model['description']}\n")
        f.write('-' * 80 + '\n\n')

# Table data
table_data = []
for model in models['data']:
    table_data.append([
        model['name'],
        model['id'],
        model['context_length'],
        f"{model['pricing']['prompt']} / {model['pricing']['completion']}"
    ])

# Table data str (simple format)
table_str = tabulate(table_data, 
                    headers=['Name', 'ID', 'Context Length', 'Pricing (prompt/completion)'], 
                    tablefmt='simple')

with open(os.path.join(current_dir, 'models_table.txt'), 'w', encoding='utf-8') as f:
    f.write(table_str)