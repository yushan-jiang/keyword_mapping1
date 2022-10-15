# -*- coding:utf-8 -*-
"""
摘要与标准术语匹配
@author:CarlSmith
@time:2022-06-03
"""

import torch
import time
from DataProcessing.CutWord import cut_singleWord
from DataProcessing.VocabCreate import abs_vocab, def_vocab


def tokenizer(text):
    tokes = []
    for tok in text.split(' '):
        tokes.append(tok)
    return tokes


# 对应预测分类结果
def predict_sentiment(net, abs_vocab_p, def_vocab_p, sentence1, sentence2):
    """sentence是词语的列表"""
    support_device = list(net.parameters())[0].device
    abs_vocab_p.set_default_index(0)
    def_vocab_p.set_default_index(0)
    se1 = torch.tensor([abs_vocab_p.__getitem__(word1) for word1 in sentence1], device=support_device)
    se2 = torch.tensor([def_vocab_p.__getitem__(word2) for word2 in sentence2], device=support_device)
    start = time.perf_counter()
    net.eval()
    with torch.no_grad():
        label = torch.round(net(se1.view((1, -1)), se2.view((1, -1))))
    end = time.perf_counter()
    print(net(se1.view((1, -1)), se2.view((1, -1))))
    # 此处修改的返回值要和java部分保持一致
    return '匹配' if label == 1 else '不匹配'


l_m = torch.load('../DataSet/LSTM_SingleWord_WiKi_100.pkl')


def result(d1, d2):
    d1 = cut_singleWord(d1)
    d2 = cut_singleWord(d2)
    d1 = tokenizer(d1)
    d2 = tokenizer(d2)
    return predict_sentiment(l_m, abs_vocab, def_vocab, d1, d2)


s1 = '神经机器翻译凭借其良好性能成为目前机器翻译的主流方法,然而,神经机器翻译编码器能否学习到充分的语义信息一直是学术上亟待探讨的问题。为了探讨该问题,该文通过利用抽象语义表示(abstract meaning representation,AMR)所包含的语义特征,分别从单词级别、句子级别两种不同的角度去分析神经机器翻译编码器究竟在多大程度上能够捕获到语义信息,并尝试利用额外的语义信息提高机器翻译性能。实验表明: 首先神经机器翻译编码器能够学习到较好的单词级和句子级语义信息;其次,当神经机器翻译的训练集规模较小时,利用额外语义信息能够提高翻译性能'
s2 = '选取的自然数作为满足某条件的证据的可能候选者'

result(s1, s2)
