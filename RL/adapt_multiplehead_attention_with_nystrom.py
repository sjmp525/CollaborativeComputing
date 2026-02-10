import torch
import torch.nn.functional as F

class AdaptiveMultiHeadAttentionWithNystrom:
    def __init__(self, input_dim, base_num_heads=4, delta=0.1, max_k=3, dropout_prob=0.1):
        self.base_num_heads = base_num_heads  # 基础的注意力头数量
        self.input_dim = input_dim  # 输入特征维度
        self.delta = delta  # 动态阈值，用于判断状态差异
        self.max_k = max_k  # 最多衍生状态数
        self.dropout = torch.nn.Dropout(p=dropout_prob)  # Dropout层，用于防止过拟合

        # 初始化每个头的查询（Q）、键值（K, V）权重矩阵，依据输入特征维度
        self.head_dims = [input_dim // (base_num_heads + i) for i in range(base_num_heads)]
        self.W_q = [torch.nn.Parameter(torch.randn(hd, input_dim)) for hd in self.head_dims]
        self.W_kv = [torch.nn.Parameter(torch.randn(hd, input_dim)) for hd in self.head_dims]

    def compute_dynamic_heads(self, s_t):
        """
        动态计算需要的注意力头数量，基于输入状态的复杂性。
        """
        complexity = torch.norm(s_t) / self.input_dim  # 计算输入状态的范数作为复杂度指标
        num_heads = min(self.base_num_heads + int(complexity * 2), len(self.head_dims))  # 动态确定头的数量
        return num_heads

    def nystrom_approximation(self, Q, K, V, m):
        """
        使用Nyström方法逼近注意力矩阵，从而减少计算复杂度
        参数:
        Q: 查询矩阵
        K: 键矩阵
        V: 值矩阵
        m: 采样的点数，用于低秩矩阵近似
        """
        n = Q.shape[0]  # 输入序列的长度
        sampled_indices = torch.randperm(n)[:m]  # 随机采样m个点
        Q_sampled = Q[sampled_indices]  # 采样后的Q
        K_sampled = K[sampled_indices]  # 采样后的K

        # 计算低秩近似的A, B矩阵
        A = torch.matmul(Q_sampled, K_sampled.T)  # 采样点之间的内积
        B = torch.matmul(Q, K_sampled.T)  # 全部点和采样点之间的内积

        # 计算A的伪逆
        A_inv = torch.linalg.pinv(A)
        attention_weights = torch.matmul(B, A_inv)  # 近似计算注意力权重
        return torch.matmul(attention_weights, V)  # 输出近似注意力结果

    def attention(self, s_t, state_history):
        """
        计算注意力机制，结合状态历史信息并通过Nyström方法进行矩阵分解
        参数:
        s_t: 当前状态
        state_history: 状态的历史序列
        """
        num_heads = self.compute_dynamic_heads(s_t)  # 动态确定头的数量
        s_derived_list = []  # 存储每个头的输出

        # 针对每个注意力头执行计算
        for i in range(num_heads):
            # 将当前状态s_t进行线性变换得到查询矩阵Q
            Q = torch.matmul(self.dropout(s_t), self.W_q[i])

            # 对历史状态进行线性变换得到键值矩阵K和V
            K = torch.stack([torch.matmul(self.dropout(s), self.W_kv[i]) for s in state_history])
            V = torch.stack([torch.matmul(self.dropout(s), self.W_kv[i]) for s in state_history])

            # 使用Nyström方法进行矩阵分解和逼近
            s_derived = self.nystrom_approximation(Q, K, V, m=3)  # 使用3个采样点进行逼近
            s_derived_list.append(s_derived)  # 保存结果

        # 将所有头的输出拼接为一个多头注意力输出
        s_derived_multihead = torch.cat(s_derived_list, dim=-1)
        return s_derived_multihead

    def dynamic_should_derive(self, s_t, s_t_i):
        """
        动态判断是否应该生成衍生状态，基于状态差异动态调整阈值
        """
        delta_dynamic = self.delta * torch.exp(-torch.norm(s_t - s_t_i))  # 动态调整delta阈值
        return torch.norm(s_t_i - s_t) > delta_dynamic  # 判断当前状态与历史状态的差异是否超过阈值

    def generate_derivative_states(self, s_t, state_sequence):
        """
        生成衍生状态序列，结合历史状态信息并通过注意力机制进行衍生
        参数:
        s_t: 当前输入状态
        state_sequence: 后续状态的历史序列
        """
        derived_states = []  # 用于存储生成的衍生状态
        count = 0  # 记录生成的衍生状态数量

        # 确保输入s_t是PyTorch张量
        if not isinstance(s_t, torch.Tensor):
            s_t = torch.tensor(s_t, dtype=torch.float32)

        # 遍历状态序列进行衍生状态生成
        for s_t_i in state_sequence:
            # 确保每个历史状态也是PyTorch张量
            if not isinstance(s_t_i, torch.Tensor):
                s_t_i = torch.tensor(s_t_i, dtype=torch.float32)

            if count >= self.max_k:  # 如果超过最大允许的衍生状态数，终止循环
                break

            # 动态判断是否生成衍生状态
            if self.dynamic_should_derive(s_t, s_t_i):
                state_history = [s_t] + [s_t_i]  # 组合当前状态和历史状态
                s_derived = self.attention(s_t, state_history)  # 通过注意力机制生成衍生状态
                derived_states.append(s_derived)  # 添加衍生状态到列表
                count += 1  # 计数+1

        return derived_states  # 返回生成的衍生状态序列
