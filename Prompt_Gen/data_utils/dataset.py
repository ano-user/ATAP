from torch.utils.data import Dataset
import json
from data_utils.vocab import *
from transformers import AutoTokenizer
from os.path import join


def load_file(filename):
    data = []
    with open(filename, "r",encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class LAMADataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = []
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.x_hs, self.x_ts = [], []
        vocab = get_vocab_by_strategy(args,tokenizer)
        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            self.x_ts.append(d['obj_label'])
            self.x_hs.append(d['sub_label'])
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]['sub_label'], self.data[i]['obj_label']

#the orginal of PLMs
# class LAMADataset_old(Dataset):
#     def __init__(self, dataset_type, data, tokenizer, args):
#         super().__init__()
#         self.args = args
#         self.data = []
#         self.dataset_type = dataset_type
#         self.tokenizer = tokenizer
#         self.x_hs, self.x_ts = [], []
#         shared_vocab = json.load(open(join(args.data_dir, '29k-vocab.json'),encoding='utf-8'))

#         vocab = shared_vocab['bert-base-cased']
#         for d in data:
#             if token_wrapper(args, d['obj_label']) not in vocab:
#                 continue
#             self.x_ts.append(d['obj_label'])
#             self.x_hs.append(d['sub_label'])
#             self.data.append(d)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         return self.data[i]['sub_label'], self.data[i]['obj_label']

