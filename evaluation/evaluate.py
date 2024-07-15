from utils import connect_db, change_sql_mode
from similarity import exact_query_match, measure_query_similarities
from data_class import Metrics, update_metrics

def evaluate_generated_queries(
    generated_queries: list[str],
    true_queries: list[dict]
) -> list[Metrics]:
    
    conn = connect_db()
    cursor = conn.cursor()
    change_sql_mode(cursor)
    
    metrics_list = []
    for query_idx in range(len(generated_queries)):
        generated_query = generated_queries[query_idx]
        true_query = true_queries[query_idx]["answer"]
        true_context = true_queries[query_idx]["context"]
        
        exact_flag = exact_query_match(
            generated_query, 
            true_query
        )
        if exact_flag:
            curr_metric = update_metrics(
                generated_query=generated_query,
                true_query=true_query,
                true_context=true_context,
                exact_flag=exact_flag,
                similarity_measure=1.,
                exec_gen_query=True
            )

            metrics_list.append(curr_metric)
            continue
            
        similarity_measure, exec_gen_query = measure_query_similarities(
            generated_query=generated_query,
            true_query=true_query,
            cursor=cursor
        )

        curr_metric = update_metrics(
            generated_query=generated_query,
            true_query=true_query,
            true_context=true_context,
            exact_flag=exact_flag,
            similarity_measure=similarity_measure,
            exec_gen_query=exec_gen_query
        )
        metrics_list.append(curr_metric)

    cursor.close()
    conn.close()

    return metrics_list

        