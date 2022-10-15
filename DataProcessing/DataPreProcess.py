"""
数据预处理
@author:CarlSmith
@time:2022-06-03
"""


import math
import torch
import torch.utils.data as Data
from DataProcessing.ObtainData import init_data
from DataProcessing.VocabCreate import def_vocab
from DataProcessing.VocabCreate import abs_vocab
from DataProcessing.utils import get_tokenized_imdb
from sklearn.model_selection import train_test_split

# cope_data:处理后的数据
# 去除不相关的列
#X是定义和摘要，Y是label
cope_data = init_data.drop(columns=['ID', 'article_id', 'technical_id'])
Y = cope_data['label']
X = cope_data.drop(columns='label')

# 数据集划分
# X_train: 训练数据
# X_odd: 需要二次划分为测试集验证集的数据
# Y_train: 训练数据标签
# Y_odd: 需要二次划分为测试集验证集的数据标签
X_train, X_odd, Y_train, Y_odd = train_test_split(X, Y, test_size=0.2, random_state=12)

# 划分测试集和验证集
X_val, X_test, Y_val, Y_test = train_test_split(X_odd, Y_odd, test_size=0.2, random_state=12)

# 将划分后的数据和对应标签相拼接
X_train['label'] = Y_train
X_val['label'] = Y_val
X_test['label'] = Y_test

# 划分后的数据处理
train_data = []
val_data = []
test_data = []

# 将训练数据处理为['摘要', '定义', '标签']的形式
for _, row in X_train.iterrows():
    mid = [row['abs'], row['define'], row['label']]
    train_data.append(mid)

# 将验证数据处理为['摘要', '定义', '标签']的形式
for _, row in X_val.iterrows():
    mid = [row['abs'], row['define'], row['label']]
    val_data.append(mid)

# 将测试数据处理为['摘要', '定义', '标签']的形式
for _, row in X_test.iterrows():
    mid = [row['abs'], row['define'], row['label']]
    test_data.append(mid)


# 计算截取长度
def cut_len(extract_data):
    min_len = len(extract_data[0])
    max_len = len(extract_data[0])
    for i in range(len(extract_data)):
        mid_len = len(extract_data[i])
        if mid_len < min_len:
            min_len = mid_len
        if mid_len > max_len:
            max_len = mid_len
    return math.ceil(max_len * 0.9)


# 摘要数据向量化
def preprocess_imdb(data):
    # tokenized_data是数据按照空格分开后的句子，是一个二维list
    abs_tokenized_data = get_tokenized_imdb(data)[0]
    def_tokenized_data = get_tokenized_imdb(data)[1]
    # 计算截取长度
    abs_max_l = cut_len(abs_tokenized_data)
    def_max_l = cut_len(def_tokenized_data)

    # 定义评论补全函数
    # 将每条评论通过截断或者补0

    def pad(x, max_l):
        return x[:max_l] if len(x) > max_l else [0] * (max_l - len(x)) + x

    # features是每个词在字典中的value
    abs_vocab.set_default_index(0)
    def_vocab.set_default_index(0)
    abs_features = torch.tensor([pad([abs_vocab.__getitem__(word) for word in words], abs_max_l)
                                 for words in abs_tokenized_data])
    def_features = torch.tensor([pad([def_vocab.__getitem__(word) for word in words], def_max_l)
                                 for words in def_tokenized_data])
    labels = torch.tensor([score for _1, _2, score in data])
    return abs_features, def_features, labels


# 创建数据迭代器
batch_size = 16
train_set = Data.TensorDataset(*preprocess_imdb(train_data))
val_set = Data.TensorDataset(*preprocess_imdb(val_data))
test_set = Data.TensorDataset(*preprocess_imdb(test_data))


# 其中每个数据集都有16个句子
# 将数据随机设置
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
val_iter = Data.DataLoader(val_set, batch_size)
test_iter = Data.DataLoader(test_set, batch_size)


def main():

    # 查看数据类型
    for abs_x, def_x, y in train_iter:
        print('abs_x', abs_x.shape, 'def_x', def_x.shape, 'y', y.shape)
        print(abs_x)
        print('-' * 100)
        print(def_x)
        print('-' * 100)
        print(y)
        break


if __name__ == "__main__":
    # main()
    # print(len(abs_vocab))
    # print(len(def_vocab))
    b= [x[1] for x in train_data]
    print(b)





