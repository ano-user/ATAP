import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re

from transformers import AutoTokenizer

import sys
# sys.path.append("P-tuning-main")
from prompt_gen.models import get_embedding_layer, create_model
from data_utils.vocab import *
from data_utils.dataset import load_file
from prompt_gen.prompt_encoder import PromptEncoder


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        # load tokenizer
        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(r'Add_tokens_PLM/PLMs/add-bert-large-cased/', use_fast=False)
        self.tokenizer_old = AutoTokenizer.from_pretrained(r'Add_tokens_PLM/PLMs/bert-large-cased/', use_fast=False)
        

        # load pre-trained model
        self.model = create_model(self.args)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
                
        self.embeddings = get_embedding_layer(self.args, self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        
        if 'gpt' in self.args.model_name:
            template = (template[0], template[1], 0)
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        
        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1] 
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None):
        # For P-tuning
        if 'gpt' not in self.args.model_name:
            # BERT-style model
            
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + self.tokenizer_old.convert_tokens_to_ids(self.tokenizer_old.tokenize(' ' + x_h))  # head entity
                    + prompt_tokens * self.template[1]
                    + [self.tokenizer.mask_token_id]  # [MASK] (tail entity)
                    + (prompt_tokens * self.template[2] if self.template[2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        
        elif 'gpt' in self.args.model_name:
            # GPT-style models
            return [prompt_tokens * self.template[0]
                    + self.tokenizer_old.convert_tokens_to_ids(self.tokenizer_old.tokenize(' ' + x_h))  # head entity
                    + prompt_tokens * self.template[1]
                    + (self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

    def forward(self, x_hs, x_ts,evaluate_type, epoch,return_candidates=False):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id

        # get embedded input
        inputs_embeds = self.embed_input(queries)

        def bert_out(x_hs, x_ts,evaluate_type,epoch):

            label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  
            labels = labels.scatter_(1, label_mask, label_ids)
            output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits
            
            # bert-base-cased/bert-large-cased:fune-tuning
            # for batch_size in range(bz):
            #     if label_ids[batch_size,0] >= 28996:
            #         logits[batch_size,label_mask[batch_size,0],:28995] = -100
            #     else:
            #         logits[batch_size,label_mask[batch_size,0],28995:] = -100

           
            # robert-base-caed/robert-large-cased:fune-tuning
            # for batch_size in range(bz):
            #     if label_ids[batch_size,0] >= 50264:
            #         logits[batch_size,label_mask[batch_size,0],:50263] = -100
            #     else:
            #         logits[batch_size,label_mask[batch_size,0],50263:] = -100
  
            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            hit3 = 0
            hit10 = 0
            top10 = []
            pred_top10 = []
            mrr = 0

            for i in range(bz):
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                num = 0
                pred_top10 = []
                temp = 0

                if evaluate_type == 'Test':
                    for pred in pred_seq:
                        if pred in self.allowed_vocab_ids:
                            temp = temp + 1
                            if pred == label_ids[i, 0] :
                                mrr = mrr + 1 / temp
                                break
                        
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        num = num + 1
                        pred_top10.append(pred)
                        if pred == label_ids[i, 0] or num > 10:
                            break

                if pred == label_ids[i, 0] and num <= 1:
                    hit1 += 1
                if pred == label_ids[i, 0] and num <= 3:
                    hit3 += 1
                if pred == label_ids[i, 0] and num <= 10:
                    hit10 += 1

            if return_candidates:
                return loss, hit1, hit3,hit10,top10,mrr
            
            if evaluate_type == 'Test':
                return loss, hit1,hit3,hit10,mrr
            
            return loss, hit1,hit3,hit10

        def gpt_out(x_hs, x_ts,evaluate_type,epoch):
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)

            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            for batch_size in range(bz):
                if label_ids[batch_size,0] >= 50256:
                    logits[batch_size,label_mask[batch_size,0],:50255] = -100
                else:
                    logits[batch_size,label_mask[batch_size,0],50255:] = -100

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            hit3 = 0
            hit10 = 0
            top10 = []
            mrr = 0

            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                temp = 0

                if evaluate_type == 'Test':
                    for pred in pred_seq:
                        if pred in self.allowed_vocab_ids:
                            temp = temp + 1
                            if pred == label_ids[i, 0] :
                                mrr = mrr + 1 / temp
                                break

                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                for j in range(10):
                    if top10[-1][j] == label_ids[i,0] and j==0:
                        hit1 += 1
                    if top10[-1][j] == label_ids[i,0] and j<=2:
                        hit3 += 1
                    if top10[-1][j] == label_ids[i,0] and j<=9:
                        hit10 += 1


            if return_candidates:
                return loss, hit1, hit3,hit10,top10,mrr
            
            if evaluate_type == 'Test':
                return loss, hit1,hit3,hit10,mrr
            
            return loss, hit1,hit3,hit10
