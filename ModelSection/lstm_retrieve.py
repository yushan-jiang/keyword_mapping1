"""
根据摘要检索术语库
@author:CarlSmith
@time:2022-06-03
"""

import torch
import pandas as pd
from ast import literal_eval
from DataProcessing.CutWord import cut_singleWord
from DataProcessing.VocabCreate import abs_vocab, def_vocab

# 加载预先处理好的术语定义向量
technical_pre_data = pd.read_csv("../DataSet/pre_vocab.csv")

# 导入模型参数
load_model = torch.load("../DataSet/LSTM_SingleWord_WiKi_100.pkl")


# 按空格拆分
def tokenizer(text):
    tokes = []
    for tok in text.split(' '):
        tokes.append(tok)
    return tokes


# 检索预测函数
def retrievePredict(abs_data):
    # 是否使用cuda
    support_device = list(load_model.parameters())[0].device
    # 拆分输入的摘要数据
    s1 = cut_singleWord(abs_data)
    s1_res = tokenizer(s1)
    # 设置默认索引
    abs_vocab.set_default_index(0)
    def_vocab.set_default_index(0)
    se1 = torch.tensor([abs_vocab.__getitem__(word1) for word1 in s1_res], device=support_device)
    # 存放结果阈值的列表
    res = []
    # 依次遍历术语表，完成检索
    print("遍历开始")
    i = 1
    for _, row in technical_pre_data.iterrows():
        # 将对应的向量从文件中提取出来，同时因为提取出来的内容是字符形式，所以需要转化为字符形式
        # 注：该部分易发生错误，慎重！！！
        se2 = torch.tensor(list(map(int, literal_eval(row['vocab']))))
        load_model.eval()
        with torch.no_grad():
            # 设置运行占用CPU内核数
            torch.set_num_threads(2)
            evaluate = load_model(se1.view((1, -1)), se2.view((1, -1)))
            if evaluate >= 0.5:
                res.append((row['standard'], float(evaluate[0][0])))
        print(i)
        i += 1
    return res


# 将检索结果排序并输出阈值最大的4个标准术语
def select_res(res):
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)
    back_res = []
    if len(res) > 4:
        for i in range(4):
            back_res.append(res[i][0])
        print(str(back_res).replace("[", "").replace("]", "").replace("\'", ""))
        return str(back_res).replace("[", "").replace("]", "").replace("\'", "")
    else:
        for i in range(len(res)):
            back_res.append(res[i][0])
        print(str(back_res).replace("[", "").replace("]", "").replace("\'", ""))
        return str(back_res).replace("[", "").replace("]", "").replace("\'", "")


# 返回检索匹配最终结果
def predict_res(abs_word):
    return select_res(retrievePredict(abs_word))
