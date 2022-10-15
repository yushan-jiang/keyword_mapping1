"""
工具文件
@author:CarlSmith
@time:2022-06-03
"""


# 文本数据预处理
def get_tokenized_imdb(data):
    """
    data: list of [string1, string2, label]
    """

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    # 获得按照空格分开后的所有词语
    return [[tokenizer(review1) for review1, review2, _ in data],
            [tokenizer(review2) for review1, review2, _ in data]]
