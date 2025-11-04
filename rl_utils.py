"""
通用工具函数
包含状态处理、数据转换等常用函数
"""

import numpy as np
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


def state_to_vector(state_dict):
    """
    将状态字典转换为向量表示
    
    参数:
        state_dict: 环境返回的状态字典
        
    返回:
        状态向量（numpy数组）
    """
    # 归一化后的位置 [norm_x, norm_y]
    normalized_pos = list(state_dict.get('normalized_agent_pos', [0, 0]))
    
    # 归一化后的距离 [norm_distance]
    normalized_distance = [state_dict.get('normalized_distance', 0)]
    
    # 归一化后的方向 [norm_dx, norm_dy]
    normalized_direction = list(state_dict.get('normalized_target_direction', [0, 0]))
    
    # 组合特征
    features = normalized_pos + normalized_distance + normalized_direction
    
    # 周围2格的扩展障碍物（4个方向×2格=8个值）[左1,左2,右1,右2,上1,上2,下1,下2]
    extended_obstacles = state_dict.get('extended_obstacles', [0] * 8)  
    features += extended_obstacles

    '''
    # 周围障碍物 [左,右,上,下]
    surrounding_obstacles = state_dict.get('surrounding_obstacles', [0, 0, 0, 0])
    features += surrounding_obstacles
    '''
    
    # 上一步动作 [-1表示无历史]
    last_action = [state_dict.get('last_action', -1)]
    features += last_action
    
    return np.array(features, dtype=np.float32)

def compute_advantage(gamma, lmbda, td_delta):
    """
    计算优势函数GAE
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)   

class ReplayBuffer:
    """经验回放池"""  
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)