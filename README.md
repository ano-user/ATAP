# ATAP:Template-Generated Commonsense Knowledge Graph Completion with Pre-Trained Language Models

> The mission of commonsense knowledge graph completion (CKGC) is to infer missing facts from known commonsense knowledge. 
%CKGC methods are categorized into two main classes: triple-based and text-based methods. 
CKGC methods can be roughly divided into two categories: triple-based methods and text-based methods.
Due to the imbalanced distribution of entities and limited structural information, triplet-based methods struggle with long-tail entities. Text-based methods alleviate this issue, but require extensive training and fine-tuning of language models, which reduces efficiency. To alleviate these problems, we propose ATAP, a framework that utilizes automatically generated continuous prompt templates combined with pre-trained language models (PLMs). Without introducing additional overhead, it makes full use of the vast knowledge base of PLMs and alleviates the long-tail problem. Moreover, ATAP uses a new prompt template training strategy, which encodes contextual structural knowledge in a commonsense knowledge graph into templates to guide the PLMs.

## Dependencies

- python==3.9.12
- pytorch==1.13.1
- transformers==4.35.0


## Running
According to the order of the paper method modules, the pre-trained language model can be downloaded from the huggingface official website.

If you have any problems running the code, please send an email to 2201776@stu.neu.edu.cn

