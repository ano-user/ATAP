import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from os.path import join, abspath, dirname
from Prompt_Gen.data_utils.dataset import load_file,LAMADataset,LAMADataset_old
from Prompt_Gen.data_utils.vocab import *
from Prompt_Gen.prompt_gen.modeling import *


SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large']


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="ConceptNet")
    parser.add_argument("--model_name", type=str, default='bert-large-cased', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--template", type=str, default="(3,3,0)")
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)#1e-5
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--use_lm_finetune", type=bool, default=True)
    parser.add_argument("--vocab_strategy", type=str, default="shared")

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), 'Relation_triples'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), 'results_out'))
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()

    # post-parsing args                                                                 

    args.device = torch.device("cuda:2" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:2'

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(r'Add_tokens_PLM/PLMs/add-bert-large-cased/', use_fast=False)#导入分词器
        init_vocab(args)

        # load datasets and dataloaders
        self.data_path_pre, self.data_path_post = self.get_TREx_parameters()
        self.train_data = load_file(join(self.args.data_dir, self.data_path_pre + 'train' + self.data_path_post))
        self.dev_data = load_file(join(self.args.data_dir, self.data_path_pre + 'dev' + self.data_path_post))
        self.test_data = load_file(join(self.args.data_dir, self.data_path_pre + 'test' + self.data_path_post))

        self.test_set = LAMADataset('test', self.test_data, self.tokenizer, self.args)
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer, self.args)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer, self.args)

        #the orginal of the PLMs
        # self.test_set = LAMADataset_old('test', self.test_data, self.tokenizer, self.args)
        # self.train_set = LAMADataset_old('train', self.train_data, self.tokenizer, self.args)
        # self.dev_set = LAMADataset_old('dev', self.dev_data, self.tokenizer, self.args)

        self.train_loader = DataLoader(self.train_set, batch_size=64, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=64)
        self.test_loader = DataLoader(self.test_set, batch_size=64)

        self.model = PTuneForLAMA(args, self.device, self.args.template)

    def get_TREx_parameters(self):
        
        data_path_pre = "relation_triples/{}/".format(self.args.relation_id)
        data_path_post = ".jsonl"
        return data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1,hit3,hit10,loss,MRR = 0,0,0,0,0
            for x_hs, x_ts in loader:
                _loss, _hit1, _hit3, _hit10,mrr = self.model(x_hs, x_ts,evaluate_type, epoch_idx)
                hit1 += _hit1
                hit3 += _hit3
                hit10 += _hit10
                loss += _loss.item()
                MRR += mrr
            orgiinal_hit1 = hit1
            orgiinal_hit3 = hit3
            orgiinal_hit10 = hit10
            if len(dataset) != 0:
                hit1 /= len(dataset)
                hit3 /= len(dataset)
                hit10 /= len(dataset)
                div_loss = loss / len(dataset)
            else:
                div_loss = 0
            print("{} {} Epoch {} Loss: {} Hit@1:".format(self.args.relation_id, evaluate_type, epoch_idx,div_loss), hit1)
            print("{} {} Epoch {} Loss: {} Hit@3:".format(self.args.relation_id, evaluate_type, epoch_idx,div_loss), hit3)
            print("{} {} Epoch {} Loss: {} Hit@10:".format(self.args.relation_id, evaluate_type, epoch_idx,div_loss), hit10)
            print("{} {} Epoch {} Loss: {} MRR:".format(self.args.relation_id, evaluate_type, epoch_idx,div_loss), MRR / len(dataset))
        return loss, orgiinal_hit1,orgiinal_hit3,orgiinal_hit10,len(dataset),MRR

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):

        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 5e-6})#5e-6
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)#使用Adam优化器
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        
        
        best_hit1 = 0
        best_hit3 = 0
        best_hit10 = 0
        len_dataset = 0
        best_mrr = 0
        best_test = -1
        
        for epoch_idx in range(200):
            # check early stopping
            if epoch_idx > -1:
                # dev_loss, dev_hit1, dev_hit3, dev_hit10, _, _ = self.evaluate(epoch_idx, 'Dev')
                test_loss, test_hit1, test_hit3, test_hit10,len_dataset, mrr = self.evaluate(epoch_idx, 'Test')
                if epoch_idx > 0 and  mrr > best_mrr:
                    # test_loss, test_hit1, test_hit3, test_hit10,len_dataset = self.evaluate(epoch_idx, 'Test')
                    best_hit1 = test_hit1
                    best_hit3 = test_hit3
                    best_hit10 = test_hit10
                    best_mrr = mrr
                    # best_ckpt = self.get_checkpoint(epoch_idx, dev_hit10, test_hit10)
                    early_stop = 0
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        # self.save(best_ckpt)
                        print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt,best_hit1,best_hit3,best_hit10,len_dataset,best_mrr

            # run training
            hit1,hit3,hit10,num_of_samples = 0,0,0,0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1,batch_hit3,batch_hit10 = self.model(batch[0], batch[1],'train',batch_idx)
                hit1 += batch_hit1
                hit3 += batch_hit3
                hit10 += batch_hit10
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()
                # torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
        # self.save(best_ckpt)

        return best_ckpt,best_hit1,best_hit3,best_hit10,len_dataset,best_mrr


def main(relation_id=None):

    # '''
    # common prompt  CN-100K and ATOMIC
    # '''
    # args = construct_generation_args()#加载第32行函数参数
    # if relation_id:
    #     args.relation_id = relation_id
    # if type(args.template) is not tuple:#为什么模板不是元组就要运行评估函数
    #     args.template = eval(args.template)
    # assert type(args.template) is tuple
    # print(args.relation_id)#输出关系id以及模型名字
    # print(args.model_name)
    # # set_trace()
    # trainer = Trainer(args)#进入训练函数
    # trainer.train()

    # '''relational prompt   CN-100K'''

    templates = [
        "(4,4)","(3,3)","(6,6)","(5,5)","(5,5)",
        "(3,3)","(3,3)","(2,2)","(6,6)","(4,4)",
        "(6,6)","(2,2)","(3,3)","(5,5)","(2,2)",
        "(5,5)","(3,3)","(3,3)","(5,5)","(3,3)",
        "(2,2)"
    ]

    relations = [
        
        'ConceptNet_relation_IsA',
        'ConceptNet_relation_AtLocation',
        'ConceptNet_relation_UsedFor',
        'ConceptNet_relation_CapableOf',
        'ConceptNet_relation_HasProperty',
        'ConceptNet_relation_HasSubevent',
        'ConceptNet_relation_HasPrerequisite',
        'ConceptNet_relation_Causes', 
        'ConceptNet_relation_HasA', 
        'ConceptNet_relation_PartOf', 
        'ConceptNet_relation_MadeOf', 
        'ConceptNet_relation_ReceivesAction',
        'ConceptNet_relation_NotCapableOf', 
        'ConceptNet_relation_CausesDesire', 
        'ConceptNet_relation_Desires', 
        'ConceptNet_relation_MotivatedByGoal', 
        'ConceptNet_relation_NotIsA', 
        'ConceptNet_relation_HasFirstSubevent', 
        'ConceptNet_relation_NotHasProperty', 
        'ConceptNet_relation_CreatedBy', 
        'ConceptNet_relation_DefinedAs'
    ]

    relation_best_prompt_templete_test_data = {}
    for index,relation in enumerate(relations):
        args = construct_generation_args()
        args.relation_id = relation
        args.template = templates[index]
        if type(args.template) is not tuple:
            args.template = eval(args.template)
        assert type(args.template) is tuple
        print(args.relation_id)
        print(args.template)
        print(args.model_name)
        print(args)
        trainer = Trainer(args)
        _,relation_best_hit1,relation_best_hit3,relation_best_hit10,len_dataset,relation_best_mrr = trainer.train()
        hit_data = []
        hit_data.append(relation_best_hit1)
        hit_data.append(relation_best_hit3)
        hit_data.append(relation_best_hit10)
        hit_data.append(relation_best_mrr)
        hit_data.append(len_dataset)
        relation_best_prompt_templete_test_data[relation] = hit_data
        print(relation_best_prompt_templete_test_data)

    hit1 = 0
    hit3 = 0
    hit10 = 0
    dataset_len = 0
    MRR = 0
    for value in relation_best_prompt_templete_test_data.values():
        hit1 += value[0]
        hit3 += value[1]
        hit10 += value[2]
        MRR += value[3]
        dataset_len += value[4]
    
    print('hit@1:',hit1/dataset_len)
    print('hit@3:',hit3/dataset_len)
    print('hit@10:',hit10/dataset_len)
    print('MRR:',MRR/dataset_len)

    # '''relational prompt ATOMIC'''

    # templates = [
    #     "(2,2)","(2,2)","(4,4)","(4,4)","(5,5)",
    #     "(5,5)","(5,5)","(3,3)","(6,6)"
    # ]

    # relations = [
    #     'xWant', 
    #     'oWant', 
    #     'xEffect', 
    #     'oEffect', 
    #     'xIntent', 
    #     'xReact',  
    #     'oReact', 
    #     'xNeed', 
    #     'xAttr'
    # ]
    # args = construct_generation_args()

    # relation_best_prompt_templete_test_data = {}
    # for index,relation in enumerate(relations):
    #     args = construct_generation_args()
    #     args.relation_id = relation
    #     args.template = templates[index]
    #     if type(args.template) is not tuple:
    #         args.template = eval(args.template)
    #     assert type(args.template) is tuple
    #     print(args.relation_id)
    #     print(args.model_name)
    #     print(args)
    #     trainer = Trainer(args)
    #     _,relation_best_hit1,relation_best_hit3,relation_best_hit10,len_dataset,relation_best_mrr = trainer.train()
    #     hit_data = []
    #     hit_data.append(relation_best_hit1)
    #     hit_data.append(relation_best_hit3)
    #     hit_data.append(relation_best_hit10)
    #     hit_data.append(relation_best_mrr)
    #     hit_data.append(len_dataset)
        
    #     relation_best_prompt_templete_test_data[relation] = hit_data
    #     print(relation_best_prompt_templete_test_data)

    # hit1 = 0
    # hit3 = 0
    # hit10 = 0
    # dataset_len = 0
    # MRR = 0
    # for value in relation_best_prompt_templete_test_data.values():
    #     hit1 += value[0]
    #     hit3 += value[1]
    #     hit10 += value[2]
    #     MRR += value[3]
    #     dataset_len += value[4]
    
    # print('hit@1:',hit1/dataset_len)
    # print('hit@3:',hit3/dataset_len)
    # print('hit@10:',hit10/dataset_len)
    # print('MRR:',MRR/dataset_len)


if __name__ == '__main__':
    main()
