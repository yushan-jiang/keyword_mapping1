"""
对数据进行分词操作
@author:CarlSmith
@time:2022-06-03

"""
import jieba
import pynlpir
import thulac
from pyhanlp import HanLP





"""
jieba分词函数
粗粒度分词
"""


def cut_wordJieBa1(word):
    cw = jieba.cut(word, cut_all=False)
    str_out = ' '.join(cw).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    return str_out


"""
jieba分词函数
细粒度分词
"""


def cut_wordJieBa2(word):
    cw = jieba.cut(word, cut_all=True)
    str_out = ' '.join(cw).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    return str_out


"""
将句子拆分为单字组合
"""


def cut_singleWord(word):
    word = word.replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    str_out = ' '.join(word)
    return str_out


"""
HanLP分词
"""


def cut_wordHanLP(word):
    # 去除词性
    HanLP.Config.ShowTermNature = False
    word = word.replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    word = HanLP.segment(word).toString()
    word = word.replace("[", "").replace("]", "").replace(",", "")
    return word


"""
中科院分词系统分词
"""


def cut_wordZKY(word):
    # 初始化分词库
    pynlpir.open()
    # 不输出词性
    words = pynlpir.segment(word, pos_tagging=False)
    pynlpir.close()
    return words


"""
清华NLP分词
"""


def cut_wordTHUC(word):
    word = word.replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    t = thulac.thulac(seg_only=True)
    word = t.cut(word, text=True)
    return word
