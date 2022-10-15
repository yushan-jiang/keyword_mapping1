"""
从数据库中获取数据
@author:CarlSmith
@time:2022-06-03
"""

import pymysql
import pandas as pd
from DataProcessing.CutWord import cut_singleWord

# 提取文献摘要数据
# TODO 连接数据库
db = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching')

# TODO 创建游标对象
cur1 = db.cursor()
cur2 = db.cursor()
# TODO 执行MySQL查询
cur1.execute(
    "select articleInformationBatch.abstract from articleInformationBatch, ReflectData where articleinformationBatch.ID = ReflectData.article_id")
cur2.execute(
    "select technical_term.definition from technical_term, ReflectData where technical_term.id = ReflectData.technical_id")

# TODO 关闭数据库
db.close()


# 使用DataFrame处理初始数据
# 定义从数据库读取数据转换成dataframe函数
def transferSQL(sql):
    conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching', charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    # 获取连接对象的描述信息
    column_describe = cursor.description
    cursor.close()
    conn.close()
    column_names = [column_describe[i][0] for i in range(len(column_describe))]
    results = pd.DataFrame([list(i) for i in results], columns=column_names)
    return results


# init_data:初始数据
# init_sql:映射表查询语句
init_sql = "select * from ReflectData"
init_data = transferSQL(init_sql)

# 在DataFrame中添加由article_id和technical_id映射得到的摘要和术语定义
# mid1, mid2用来存放从数据库中取出的数据
mid1 = []
mid2 = []
for res1 in cur1:
    mid1.append(res1[0])
for res2 in cur2:
    mid2.append(res2[0])

init_data['abs'] = mid1
init_data['define'] = mid2

# 对文献摘要和标准术语进行分词
init_data['abs'] = init_data['abs'].apply(cut_singleWord)
init_data['define'] = init_data['define'].apply(cut_singleWord)



# symbol_data: 用来构建词典的数据
symbol_data = []

for _, row in init_data.iterrows():
    mid = [row['abs'], row['define'], row['label']]
    symbol_data.append(mid)
