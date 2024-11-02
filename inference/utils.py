import os 
import json
from jinja2 import Template
from prompts import USER_PROMPT
import re

def save_queries(model_id, generated_queries, few_shot):
    # Ensure the directory exists
    if not os.path.exists("./generated_queries"):
        os.makedirs("./generated_queries")
    
    # Replace slashes in model_id to prevent directory errors
    safe_model_id = model_id.replace('/', '_')
    
    # Create the file path
    file_path = f'./generated_queries/query_{safe_model_id}_few_shot={few_shot}.json'
    
    # Save the generated queries to a file
    with open(file_path, 'w') as f:
        json.dump(generated_queries, f)

    print(f"Queries saved to {file_path}")


def extract_sql_query(text):
    # Define patterns to identify and extract the SQL query
    query_pattern = r"```[\s\S]*?```|SELECT[\s\S]*?;"
    code_block_pattern = r"```[\s\S]*?```"

    # Find all matches for the SQL query pattern
    matches = re.findall(query_pattern, text)

    if not matches:
        return None  # No SQL query found

    # Prioritize finding a query within triple backticks
    for match in matches:
        if re.match(code_block_pattern, match):
            query = match.strip('`')
            return query.strip()

    # If no triple backtick query is found, return the first matched query
    return matches[0].strip()
