import json
import pandas as pd

df = pd.read_csv('../data/ConceptNet/test.txt',sep='\t',header=None)

all_relation = list(set(df[0]))

for i in range(len(all_relation)):

    file_name = '../CKG_Triples_class/train_relation_data/train_relation_' + all_relation[i] + '.txt'
    with open(file_name,'r',encoding='utf-8') as f:
        contents = f.readlines()
        out_file_name = 'relation_triples/ConceptNet_relation_' + all_relation[i] + '/train.jsonl'
        for content in contents:
            parts = content.split('	')
            head = parts[1]
            relation = parts[0]
            tail = parts[2].strip()
            msg = {"obj_label":tail,"sub_label":head,"predicate":relation}
            msg = json.dumps(msg)
            with open(out_file_name,'a',encoding='utf-8') as file:
                file.write(msg + '\n')
        file.close()
    f.close()

    file_name = '../CKG_Triples_class/dev_relation_data/dev_relation_' + all_relation[i] + '.txt'
    with open(file_name,'r',encoding='utf-8') as f:
        contents = f.readlines()
        out_file_name = 'relation_triples/ConceptNet_relation_' + all_relation[i] + '/dev.jsonl'
        for content in contents:
            parts = content.split('	')
            head = parts[1]
            relation = parts[0]
            tail = parts[2].strip()
            msg = {"obj_label":tail,"sub_label":head,"predicate":relation}
            msg = json.dumps(msg)
            with open(out_file_name,'a',encoding='utf-8') as file:
                file.write(msg + '\n')
        file.close()
    f.close()

    file_name = '../CKG_Triples_class/test_relation_data/test_relation_' + all_relation[i] + '.txt'
    with open(file_name,'r',encoding='utf-8') as f:
        contents = f.readlines()
        out_file_name = 'relation_triples/ConceptNet_relation_' + all_relation[i] + '/test.jsonl'
        for content in contents:
            parts = content.split('	')
            head = parts[1]
            relation = parts[0]
            tail = parts[2].strip()
            msg = {"obj_label":tail,"sub_label":head,"predicate":relation}
            msg = json.dumps(msg)
            with open(out_file_name,'a',encoding='utf-8') as file:
                file.write(msg + '\n')
        file.close()
    f.close()
