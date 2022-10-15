# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig

class BertModel(nn.Module):
    def __init__(self):
        super(BertModel,self).__init__()
        # self.bert = BertModel.from_pretrained('./bert_pretrain')
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度


    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks,
                              token_type_ids=batch_seq_segments, labels = labels)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelTest(nn.Module):
    def __init__(self):
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained('bert_pretrain/bert_config.json')
        self.bert = BertForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks,
                              token_type_ids=batch_seq_segments, labels = labels)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
