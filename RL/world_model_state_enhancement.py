import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class WorldModelPredictor(nn.Module):
    """世界模型预测器，用于预测未来状态"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 1)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一个状态"""
        input_tensor = torch.cat([state, action], dim=-1)
        encoded = self.encoder(input_tensor)
        next_state_pred = self.decoder(encoded)
        return next_state_pred


class UncertaintyEstimator(nn.Module):
    """不确定性估计器，用于评估状态预测的可靠性"""

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回状态预测的不确定性分数 (0-1)"""
        return self.uncertainty_net(state)


class WorldModelStateEnhancer(nn.Module):
    """基于世界模型的状态增强器，通过预测未来状态提升状态表示"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 history_length: int = 10,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        # 组件初始化
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        # 1. 世界模型预测器
        self.world_model = WorldModelPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )

        # 2. 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            state_dim=state_dim,
            hidden_dim=hidden_dim // 2
        )

        # 3. 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

        # 历史状态缓冲区
        self.register_buffer('state_history', torch.zeros(history_length, state_dim))
        self.register_buffer('action_history', torch.zeros(history_length, action_dim))
        self.history_ptr = 0
        self.history_filled = False

    def update_history(self, state: torch.Tensor, action: torch.Tensor) -> None:
        """更新历史状态和动作缓冲区"""
        self.state_history[self.history_ptr] = state.detach()
        self.action_history[self.history_ptr] = action.detach()
        self.history_ptr = (self.history_ptr + 1) % self.history_length
        if self.history_ptr == 0 and not self.history_filled:
            self.history_filled = True

    def predict_future_states(self, state: torch.Tensor, action: torch.Tensor, steps: int = 3) -> List[torch.Tensor]:
        """预测未来多个时间步的状态"""
        future_states = []
        current_state = state

        for _ in range(steps):
            next_state = self.world_model(current_state, action)
            future_states.append(next_state)
            current_state = next_state  # 递归预测

        return future_states

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        输入:
            state [batch_size, state_dim]
            action [batch_size, action_dim]
        输出:
            enhanced_state [batch_size, state_dim]
        """
        batch_size = state.size(0)

        # 1. 更新历史状态和动作
        if batch_size == 1:  # 单样本情况
            self.update_history(state[0], action[0])

        # 2. 使用世界模型预测下一个状态
        predicted_state = self.world_model(state, action)

        # 3. 估计当前状态的不确定性
        uncertainty = self.uncertainty_estimator(state)

        # 4. 预测未来多个时间步的状态
        future_states = self.predict_future_states(state, action, steps=3)

        # 5. 计算预测的平均状态作为附加特征
        avg_future_state = torch.mean(torch.stack(future_states), dim=0)

        # 6. 特征融合
        combined_features = torch.cat([
            state,
            predicted_state,
            avg_future_state,
            uncertainty
        ], dim=1)

        enhanced_state = self.fusion_layer(combined_features)

        return enhanced_state

