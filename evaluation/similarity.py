from typing import Tuple
import re

def measure_query_similarities(
        generated_query: str,
        true_query: str, 
        cursor, 
        exec_query: bool =True
) -> Tuple[float, float, bool]:
    try:
        cursor.execute(generated_query)
        gen_result = cursor.fetchall()
        cursor.nextset()
    except Exception as e:
        print(f"Failed to execute query: {generated_query}\nERROR: {e}\n")
        gen_result = []
        exec_query = False
        
    try:  
        cursor.execute(true_query)
        true_result = cursor.fetchall()
        cursor.nextset()
    except Exception as e:
        print(f"Failed to execute query: {true_query}\nERROR: {e}\n")
        true_result = []
    query_similarity = result_set_similarity(gen_result=gen_result, true_result=true_result)
    str_similarity = string_similarity(generated_query, true_query)

    return str_similarity, query_similarity, exec_query

def string_similarity(gen_query, true_query):
    gen_query = normalized_string(gen_query)
    true_query = normalized_string(true_query)
    gen_set = set(gen_query.split())
    true_set = set(true_query.split())
    if not true_set:
        return 1.0 if not gen_set else 0.0
    matches = gen_set.intersection(true_set)
    similarity_score = len(matches) / len(true_set)
    return similarity_score

def result_set_similarity(
        gen_result: str, 
        true_result: str
) -> float:
    gen_set = set(gen_result)
    true_set = set(true_result)
    if not true_set:
        return 1.0 if not gen_set else 0.0

    # Calculate intersection (matches) and union
    matches = gen_set.intersection(true_set)
    similarity_score = len(matches) / len(true_set)
    
    return similarity_score

def normalized_string(query):
    if query is None:
        return ""
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    # Convert to lowercase for case-insensitive comparison
    query = query.lower()
    query = query.replace(";","")
    query = query.replace("\"", "'")
    return query


def exact_query_match(
    generated_query: str,
    true_query: str
) -> bool:
    normal_gen = normalized_string(generated_query)
    normal_true = normalized_string(true_query)
    if normal_gen == normal_true:
        return True
    return False
