"""
模型测试集评估
@author:CarlSmith
@time:2022-06-03
"""

import torch
from DataProcessing.DataPreProcess import test_iter
from sklearn.metrics import classification_report


# 模型导入
model_load = torch.load('../DataSet/LSTM_SingleWord_WiKi_100.pkl')


# 测试集合准确率计算
def AccuracyCalculate(test_data, model_info, support_device=None):
    # 判定是否使用加速设备，如果没有指定设备，则使用默认的模型设备
    if support_device is None and isinstance(model_info, torch.nn.Module):
        support_device = list(model_info.parameters())[0].device
    y_l = []
    y_h = []
    # 准确率初始化
    acc_num = 0
    n = 0
    with torch.no_grad():
        for d1, d2, d3 in test_data:
            if isinstance(model_info, torch.nn.Module):
                # 启动评估模式
                model_info.eval()
                # 测试数据的标签
                test_label = d3
                for i in d3:
                    y_l.append(i.int())
                # 根据当前训练轮数的训练参数来使用测试集对模型效果进行拟合的结果
                test_hat = model_info(d1.to(support_device), d2.to(support_device))
                mid = torch.round(test_hat[:, 0])
                for h in torch.round(test_hat[:, 0]):
                    y_h.append(h.int())
                # 准确率求和
                acc_num += float(torch.sum(torch.round(test_hat[:, 0]) == test_label))
                model_info.train()
            n += d3.shape[0]
    print(classification_report(y_l, y_h))


# 输出测试结果
AccuracyCalculate(test_iter, model_load)

