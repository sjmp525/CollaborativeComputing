from config import server_list
from config import local_server_list


def get_client_app_port(ip_address, model_name):
    """
    根据客户端的 ip地址 和 配置的模型名称 获取部署的模型的应用的端口
    :param ip_address: 客户端的ip地址
    :param model_name: 模型名称
    :return:
    """
    # 在服务器列表中查找对应的IP地址
    for server in server_list:
        if server['ip'] == ip_address:
            # 查找应用程序名称对应的端口
            return server['application'].get(model_name)
    # 如果没有找到IP地址或应用程序名称，返回None
    return None


def get_client_app_port_by_name(name, model_name):
    """
    根据客户端的 ip地址 和 配置的模型名称 获取部署的模型的应用的端口
    :param name: 客户端的名称
    :param model_name: 模型名称
    :return:
    """
    # 在服务器列表中查找对应的IP地址
    for server in local_server_list:
        if server['name'] == name:
            # 查找应用程序名称对应的端口
            return server['ip'], server['application'].get(model_name)
    # 如果没有找到IP地址或应用程序名称，返回None
    return None