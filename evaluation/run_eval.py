from utils import load_generated_queries, load_true_queries
from evaluate import evaluate_generated_queries
from data_class import export_to_csv

def run_evaluation(path_to_generated_queries: str) -> list[str]:
    generated_queries = load_generated_queries(file_path=path_to_generated_queries)


    true_queries = load_true_queries()
   
    res = evaluate_generated_queries(generated_queries=generated_queries, true_queries=true_queries)
    # calc_final_metrics()
    export_to_csv(res, path_to_generated_queries)
    print("#"*40)
    print("DONE")
    print(res)
    return res

if __name__ == "__main__":
    run_evaluation("../results/test_data.json")