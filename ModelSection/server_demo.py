"""
thrift java和python的连接
@author:CarlSmith
@time:2022-06-03
"""

import jieba
from ThriftAPI import ModelPredict
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from ModelSection.lstm_predict import result
from ModelSection.lstm_retrieve import predict_res


class ModelAPI:
    def __init__(self):
        pass

    @staticmethod
    def cut_data(data):
        text = jieba.cut(data, cut_all=False)
        str_out = ' '.join(text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
            .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
            .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
            .replace('’', '')
        return str_out

    # 匹配函数
    def predictMatch(self, abs_word, def_word):
        return result(abs_word, def_word)

    # 检索函数
    def predictSearch(self, abs_word):
        return predict_res(abs_word)


handler = ModelAPI()
processor = ModelPredict.Processor(handler)
# 服务器端套接字管理
transport = TSocket.TServerSocket("127.0.0.1", 8083)
# 传输方式，使用buffer
t_factory = TTransport.TBufferedTransportFactory()
# 传输的数据类型：二进制
p_factory = TBinaryProtocol.TBinaryProtocolFactory()
# 创建一个thrift 服务~
server = TServer.TThreadPoolServer(processor, transport, t_factory, p_factory)


def main():
    server.serve()


if __name__ == "__main__":
    print("Starting thrift server in python...")
    main()
    print("done!")
