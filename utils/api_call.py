import json
import sqlite3
import numpy as np

"""

sqlitebiter file ../data/dialog-bAbI-tasks/dialog-babi-kb-all.json



"""

def dump_db_file(kb_path='../data/dialog-bAbI-tasks/dialog-babi-kb-all.txt'):
    f_kb = open(kb_path, 'r')
    f_kb = [line.strip() for line in f_kb.readlines()]
    
    kb_dict = {}
    for line in f_kb:
        x, value = line.split('\t')
        _, name, slot = x.split()
        if name not in kb_dict.keys():
            kb_dict[name] = {}
        kb_dict[name][slot] = value

    kb_dict_list = []
    for k, v in kb_dict.items():
        temp = {"name": k}
        for kk, vv in v.items():
            temp[kk] = vv
        kb_dict_list.append(temp)

    with open('../data/dialog-bAbI-tasks/dialog-babi-kb-all.json', 'w') as outfile:
        json.dump(kb_dict_list, outfile, indent=4)


# loading databases
db = "../data/dialog-bAbI-tasks/out.sqlite"#dialog-babi-kb-all.json"
conn = sqlite3.connect(db)
db_c = conn.cursor()

def get_api_call(api):
    sql_query = "select * from dialog_babi_kb_all where"

    slot_type = ["R_cuisine", "R_location", "R_number", "R_price"]
    for i, v in enumerate(api.split()):
        val = v.replace("'", "''")
        if i != 0:
            sql_query += r" and "
        sql_query += r" " + slot_type[i] + "=" + r"'" + val + r"'" 

    print("sql_query", sql_query)

    results = db_c.execute(sql_query).fetchall()
    print(results)


if __name__=="__main__":
    dump_db_file()
    get_api_call("spanish rome six expensive")
