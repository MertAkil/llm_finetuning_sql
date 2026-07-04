from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "test_data.jsonl"
FEW_SHOT_PATH = REPO_ROOT / "data" / "few_shot.jsonl"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "train" / "checkpoints_lora"

PROMPT_TEMPLATE = "[INST] Write a SQL query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {schema} Question: {question}[/INST] SQL query: ``` "

PROMPT_TEMPLATE_FS = "[INST] Write a SQL query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {schema} Question: {question} Example 1: Question - List the document ids for any documents with the status code done and the type code paper. Schema - create table documents (document_id varchar, document_status_code varchar, document_type_code varchar, PRIMARY KEY (document_id)); SQL query - select document_id from documents where document_status_code = \"done\" and document_type_code = \"paper\" Example 2: Question - Find the number of characteristics that the product \"flax\" has. Schema - create table characteristics (characteristic_id varchar, PRIMARY KEY (characteristic_id)); create table products (product_id varchar, product_name varchar, PRIMARY KEY (product_id)); create table product_characteristics (product_id varchar, characteristic_id varchar, PRIMARY KEY (product_id)) SQL query - select count(*) from products as t1 join product_characteristics as t2 on t1.product_id = t2.product_id join characteristics as t3 on t2.characteristic_id = t3.characteristic_id where t1.product_name = \"flax\" Example 3: Question - List all pilot names in ascending alphabetical order. Schema - create table pilot (name varchar, PRIMARY KEY (name)) SQL query - select name from pilot order by name [/INST] SQL query: ``` "

PROMPT_TEMPLATE_FS_s = PROMPT_TEMPLATE_FS


model_to_hf_name = {
    "llama2": "NousResearch/Llama-2-7b-hf",
    "codellama": "codellama/CodeLlama-7b-instruct-hf",
    "tinyllama": "TinyLlama/TinyLlama_v1.1"
}
