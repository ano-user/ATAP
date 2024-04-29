import pandas as pd

df_train = pd.read_csv('../data/ConceptNet/train.txt',sep='\t',header=None)

all_relation = list(set(df_train[0]))

for i in range(len(all_relation)):
    with open('../data/ConceptNet/train.txt','r',encoding='utf-8') as f:
        file_name = 'train_relation_data/train_relation_' + all_relation[i] + '.txt'
        contents = f.readlines()
        for content in contents:
            if all_relation[i] == content.split('	')[0]: 
                with open(file_name,'a',encoding='utf-8') as file:
                    file.write(content)
        file.close()



df_dev = pd.read_csv('../data/ConceptNet/valid.txt',sep='\t',header=None)

all_relation = list(set(df_dev[0]))

for i in range(len(all_relation)):
    with open('../data/ConceptNet/valid.txt','r',encoding='utf-8') as f:
        file_name = 'dev_relation_data/dev_relation_' + all_relation[i] + '.txt'
        contents = f.readlines()
        for content in contents:
            if all_relation[i] == content.split('	')[0]: 
                with open(file_name,'a',encoding='utf-8') as file:
                    file.write(content)
        file.close()


df_test = pd.read_csv('../data/ConceptNet/test.txt',sep='\t',header=None)

all_relation = list(set(df_test[0]))

for i in range(len(all_relation)):
    with open('../data/ConceptNet/test.txt','r',encoding='utf-8') as f:
        file_name = 'test_relation_data/test_relation_' + all_relation[i] + '.txt'
        contents = f.readlines()
        for content in contents:
            if all_relation[i] == content.split('	')[0]: 
                with open(file_name,'a',encoding='utf-8') as file:
                    file.write(content)
        file.close()


            