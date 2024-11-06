import re
import json
from typing import Any
import pandas as pd
import mysql.connector
import os

# make connector as environment vars
def connect_db() -> mysql.connector.connection_cext.CMySQLConnection:
    connection = mysql.connector.connect(
    host="localhost",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database="master_dummy"
    )

    return connection


def load_generated_queries(file_path: str) -> list[str]:
    df = pd.read_csv(file_path)
    return list(df["Results"])
    # with open("./../data/test_data.json", "r") as stream:
    #     file = json.load(stream)

    # print("Loaded generated data")
    # # with open(file_path, "r") as stream:
    # #     file = json.load(stream)
    # return [entry["answer"] for entry in file]

def load_true_queries() -> list[str]:
    data = []
    with open("./../data/test_data.jsonl", "r") as stream:
        for line in stream:
            # Parse each line as JSON and append to list
            data.append(json.loads(line.strip()))
    print("Loaded true data")
    return data

def change_sql_mode(cursor: mysql.connector.cursor_cext.CMySQLCursor) -> None:
    change_full_group_sql = "SET GLOBAL sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));"
    cursor.execute(change_full_group_sql)



def check_for_join(query: str) -> bool:
    join_pattern = re.compile(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN)\b', re.IGNORECASE)
    if join_pattern.search(query):
        return True
    return False

def check_for_tables(context:str) -> bool:
    # Use a regular expression to find all standalone occurrences of 'create table'
    create_table_count = len(re.findall(r'\bcreate table\b', context, re.IGNORECASE))
    return create_table_count > 1

