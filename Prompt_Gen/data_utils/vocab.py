import json
from os.path import join


shared_vocab = None
def init_vocab(args):
    global shared_vocab
    shared_vocab = json.load(open(join(args.data_dir, 'PLMs-vocab.json'),encoding='utf-8'))

def token_wrapper(args, token):
    if 'roberta' in args.model_name or 'gpt' in args.model_name:
        return 'Ġ' + token
    else:
        return token

def get_vocab(model_name, strategy):
    
    if 'gpt' in model_name:
        return shared_vocab['gpt2-add-tokens']
    elif 'roberta' in model_name or 'megatron' in model_name:
        return shared_vocab['gpt2-add-tokens']
    elif model_name == 'bert-base-cased' or model_name == 'bert-large-cased':
        return shared_vocab['add_tokens']
    else:
        assert model_name in shared_vocab
        return shared_vocab[model_name]

def get_vocab_by_strategy(args, tokenizer):

    return get_vocab(args.model_name, args.vocab_strategy)


