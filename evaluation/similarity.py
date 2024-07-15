from typing import Tuple

def measure_query_similarities(
        generated_query: str, 
        true_query: str, 
        cursor, 
        exec_query: bool =True
) -> Tuple[float, bool]:
    try:
        cursor.execute(generated_query)
        gen_result = cursor.fetchall()
    except Exception as e:
        print(f"Failed to execute query: {true_query}\nERROR: {e}\n")
        gen_result = None
        exec_query = False
        
    try:  
        cursor.execute(true_query)
        true_result = cursor.fetchall()
    except Exception as e:
        # Should not happen as tested already
        print(f"Failed to execute query: {true_query}\nERROR: {e}\n")
        true_result = None
        
    similarity = result_set_similarity(gen_result=gen_result, true_result=true_result)

    return similarity, exec_query

def result_set_similarity(
        gen_result: str, 
        true_result: str
) -> float:
    pass

def exact_query_match(
    generated_query: str,
    true_query: str
) -> bool:
    if generated_query == true_query:
        return True
    return False
