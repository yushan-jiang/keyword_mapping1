"""
词典构建文件
@author:CarlSmith
@time:2022-06-03
"""


import collections
import torchtext.vocab as Vocab
from DataProcessing.utils import get_tokenized_imdb
from DataProcessing.ObtainData import symbol_data


# 获取数据字典
def get_vocab_imdb(tokenized_data):
    # counter是这个数据里所有单词的出现次数
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 返回一个vocab，获取counter里单词数量大于等于1的数据
    return Vocab.vocab(counter, min_freq=1)


# 获取数据字典
abs_vocab = get_vocab_imdb(get_tokenized_imdb(symbol_data)[0])
def_vocab = get_vocab_imdb(get_tokenized_imdb(symbol_data)[1])
