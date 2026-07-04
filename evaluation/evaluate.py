from typing import Optional

try:
    from .utils import connect_db, change_sql_mode
    from .similarity import exact_query_match, measure_query_similarities, string_similarity
    from .data_class import Metrics, update_metrics
except ImportError:
    from utils import connect_db, change_sql_mode
    from similarity import exact_query_match, measure_query_similarities, string_similarity
    from data_class import Metrics, update_metrics

def evaluate_generated_queries(
    generated_queries: list[str],
    true_queries: list[dict],
    execute: bool = True,
    db_config: Optional[dict] = None,
) -> list[Metrics]:
    if len(generated_queries) != len(true_queries):
        raise ValueError(
            "Generated and reference query counts differ: "
            f"{len(generated_queries)} generated vs {len(true_queries)} reference."
        )

    conn = None
    cursor = None
    if execute:
        conn = connect_db(**(db_config or {}))
        cursor = conn.cursor()
        change_sql_mode(cursor)

    metrics_list = []
    for query_idx in range(len(generated_queries)):
        generated_query = generated_queries[query_idx]
        true_query = true_queries[query_idx]["query"]
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
                query_similarity_measure=1.,
                string_similarity_measure=1.,
                exec_gen_query=execute
            )

            metrics_list.append(curr_metric)
            continue

        if not execute:
            curr_metric = update_metrics(
                generated_query=generated_query,
                true_query=true_query,
                true_context=true_context,
                exact_flag=exact_flag,
                query_similarity_measure=0.,
                string_similarity_measure=string_similarity(generated_query, true_query),
                exec_gen_query=False
            )
            metrics_list.append(curr_metric)
            continue

        string_similarity_measure, query_similarity_measure, exec_gen_query = measure_query_similarities(
            generated_query=generated_query,
            true_query=true_query,
            cursor=cursor
        )

        curr_metric = update_metrics(
            generated_query=generated_query,
            true_query=true_query,
            true_context=true_context,
            exact_flag=exact_flag,
            query_similarity_measure=query_similarity_measure,
            string_similarity_measure = string_similarity_measure,
            exec_gen_query=exec_gen_query
        )
        metrics_list.append(curr_metric)

        if not exec_gen_query:
            print(generated_query)

    if cursor is not None:
        cursor.close()
    if conn is not None:
        conn.close()

    return metrics_list


