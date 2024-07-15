# llm_finetuning_sql

## TODO
- Write the functions from the notebook as legit util functions
- DO:
    - Search SQL commands where a specific pattern is expected (LIKE, EQUALS, etc.) and insert dummy data for that table with the expected pattern at least 2 times
    - If no pattern expected insert some dummy data (strings, random numbers). Each row should have about 1000 rows
    - Run every query in the dataset to see if empty queries are returned. 
    - If empty pattern, then fill
    - Repeat until no empty query response
        - LIKE operator: get the remainder of the string after LIKE and create such a string for column named before LIKE 
        - <> (String) operator: can be dismissed as no match is expected to get queries --> no need to cover
        - < : random 100 rows with up to 10 and random 100 rows up to 1000 --> dismis most of the time it is count
        - >:  --> dismis most of the time it is count
        - IN dismiss
        - where with = : same as LIKE operator --> covered
        - AND : get both columns and expected values. Create that --> covered
        - OR : same as and --> covered
- Prepare evaluation pipeline
    - STEPS: input -> inference -> check with expected output -> Is equal? PERFECT
                                                                 Is not equal? Run generated and true query on database and evaluate the query response as equal or not equal