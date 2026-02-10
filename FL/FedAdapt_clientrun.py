import torch  # 导入PyTorch库
import socket  # 导入socket库,用于网络通信
import time  # 导入time库,用于计时
import multiprocessing  # 导入multiprocessing库,用于多进程处理
import os  # 导入os库,用于执行系统命令
import argparse  # 导入argparse库,用于解析命令行参数
import threading  # 导入threading库,用于多线程处理
import pickle  # 导入pickle库,用于序列化对象

import logging  # 导入logging库,用于日志记录

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 创建Logger对象

import sys  # 导入sys库

sys.path.append('../')  # 将上级目录添加到Python模块搜索路径
from Client import Client  # 从Client模块导入Client类
import config  # 导入配置模块
import utils  # 导入工具模块

parser = argparse.ArgumentParser()  # 创建参数解析器
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type=utils.str2bool, default=False)  # 添加offload参数
args = parser.parse_args()  # 解析命令行参数

ip_address = config.HOST2IP[socket.gethostname()]  # 获取本机IP地址
index = config.CLIENTS_CONFIG[ip_address]  # 获取客户端索引
datalen = config.N / config.K  # 计算每个客户端的数据长度
split_layer = config.split_layer[index]  # 获取分割层
LR = config.LR  # 获取学习率

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, config.model_name, split_layer)  # 创建客户端对象

offload = args.offload  # 获取offload参数值
first = True  # 首次初始化控制标志
client.initialize(split_layer, offload, first, LR)  # 初始化客户端
first = False

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()  # 获取CPU核心数
trainloader, classes = utils.get_local_dataloader(index, cpu_count)  # 获取本地数据加载器

if offload:
    logger.info('FedAdapt Training')  # 如果开启offload,则执行FedAdapt训练
else:
    logger.info('Classic FL Training')  # 否则执行经典联邦学习训练

res = {}  # 创建字典，用于存储训练结果
res['training_time'] = []  # 初始化结果记录列表


flag = False  # 带宽控制标志
for r in range(config.R):  # 训练轮数循环
    logger.info('====================================>')
    logger.info('ROUND: {} START'.format(r))

    # 网络带宽变化
    if socket.gethostname() == 'client1':
        if r == 50 and flag == False:  # 从下一轮开始
            cmd = "sudo tc qdisc add dev zjj root tbf rate 5mbit latency 10ms burst 1600"
            pass  # Jetson需要重建Linux内核
        if r == 60 and flag == True:  # 从下一轮开始
            cmd = "sudo tc qdisc del dev zjj root"
            pass

    if socket.gethostname() == 'client2':
        if r == 60 and flag == False:  # 从下一轮开始
            cmd = "sudo tc qdisc add dev zjj root tbf rate 5mbit latency 10ms burst 1600"
            print(cmd)
            os.system(cmd)  # 执行系统命令,限制网络带宽
            flag = True
        if r == 70 and flag == True:  # 从下一轮开始
            cmd = "sudo tc qdisc del dev zjj root"
            print(cmd)
            os.system(cmd)  # 执行系统命令,取消网络带宽限制
            flag = False

    if socket.gethostname() == 'client3':
        if r == 70 and flag == False:  # 从下一轮开始
            cmd = "sudo tc qdisc add dev zjj root tbf rate 5mbit latency 10ms burst 1600"
            print(cmd)
            os.system(cmd)
            flag = True
        if r == 80 and flag == True:  # 从下一轮开始
            cmd = "sudo tc qdisc del dev zjj root"
            print(cmd)
            os.system(cmd)
            flag = False

    # if socket.gethostname() == 'fedadapt_node4':
    #     if r == 80 and flag == False:  # 从下一轮开始
    #         cmd = "sudo tc qdisc add dev zjj root tbf rate 5mbit latency 10ms burst 1600"
    #         print(cmd)
    #         os.system(cmd)
    #         flag = True
    #     if r == 90 and flag == True:  # 从下一轮开始
    #         cmd = "sudo tc qdisc del dev zjj root"
    #         print(cmd)
    #         os.system(cmd)
    #         flag = False
    #
    # if socket.gethostname() == 'fedadapt_node5':
    #     if r == 90 and flag == False:  # 从下一轮开始
    #         cmd = "sudo tc qdisc add dev zjj root tbf rate 5mbit latency 10ms burst 1600"
    #         print(cmd)
    #         os.system(cmd)
    #         flag = True
    #     if r == 100 and flag == True:  # 从下一轮开始
    #         cmd = "sudo tc qdisc del dev zjj root"
    #         print(cmd)
    #         os.system(cmd)
    #         flag = False

    training_time = client.train(trainloader)  # 客户端训练
    logger.info('ROUND: {} END'.format(r))
    # logger.info('Training time: ' + str(training_time))  # 记录训练时间
    res['training_time'].append(training_time)  # 记录训练时间

    with open('FedAdapt_res_client.pkl', 'wb') as f:  # 将结果保存到文件
        pickle.dump(res, f)

    logger.info('==> Waiting for aggregration')
    client.upload()  # 上传模型参数到服务器

    logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
    s_time_rebuild = time.time()
    if offload:
        config.split_layer = client.recv_msg(client.sock)[1]  # 接收服务器发送的分割层信息

    if r > 49:
        LR = config.LR * 0.1  # 如果训练轮数大于49,则降低学习率

    client.reinitialize(config.split_layer[index], offload, first, LR)  # 重新初始化客户端
    e_time_rebuild = time.time()
    logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))  # 记录重建时间
    logger.info('==> Reinitialization Finish')