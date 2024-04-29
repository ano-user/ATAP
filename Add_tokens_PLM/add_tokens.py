from transformers import BertForMaskedLM,BertTokenizer

all_entities = []
with open('../data/ConceptNet/train.txt','r',encoding='utf-8') as f:
    triples = f.readlines()
    
    for triple in triples:
        relation,head,tail = triple.strip().split('	')
        if head not in all_entities:
            all_entities.append(head)
        if tail not in all_entities:
            all_entities.append(tail)
with open('../data/ConceptNet/test.txt','r',encoding='utf-8') as f:
    triples = f.readlines()
    
    for triple in triples:
        relation,head,tail = triple.strip().split('	')
        if head not in all_entities:
            all_entities.append(head)
        if tail not in all_entities:
            all_entities.append(tail)
with open('../data/ConceptNet/valid.txt','r',encoding='utf-8') as f:
    triples = f.readlines()
    for triple in triples:
        relation,head,tail = triple.strip().split('	')
        if head not in all_entities:
            all_entities.append(head)
        if tail not in all_entities:
            all_entities.append(tail)


model = r"PLMs/bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model)
model = BertForMaskedLM.from_pretrained(model)
num_added_toks = tokenizer.add_tokens(all_entities)
model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained(r"PLMs/add-bert-base-cased")
model.save_pretrained(r"PLMs/add-bert-base-cased")