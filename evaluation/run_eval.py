import argparse
from utils import load_generated_queries, load_true_queries
from evaluate import evaluate_generated_queries
from data_class import export_to_csv

def parse_args():
    parser = argparse.ArgumentParser(description="Model and Few-Shot Argument Parsing")
    parser.add_argument(
        "--file_path",
        type=str,
        required=False,
        default="../results/test_data.json",
        help="choose file with generated queries"
    )    
    
    return parser.parse_args()


def run_evaluation(path_to_generated_queries: str) -> list[str]:
    generated_queries = load_generated_queries(file_path=path_to_generated_queries)


    true_queries = load_true_queries()
   
    res = evaluate_generated_queries(generated_queries=generated_queries, true_queries=true_queries)
    # calc_final_metrics()
    export_to_csv(res, path_to_generated_queries)
    print("#"*40)
    print("DONE")
    # print(res)
    return res

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.file_path)