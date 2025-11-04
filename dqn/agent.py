"""
DQN智能体模块
包含DQN算法的核心实现
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl_utils import Qnet


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, tensorboard_log_dir=None, dqn_type='VanillaDQN'):
        """
        初始化DQN智能体
        
        参数:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: epsilon-贪婪策略参数
            target_update: 目标网络更新频率
            device: 计算设备(CPU/GPU)
            tensorboard_log_dir: TensorBoard日志目录
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0  # 计数器，记录更新次数
        self.training_step = 0
        self.dqn_type = dqn_type  # DQN类型：VanillaDQN或DoubleDQN
        
        # 初始化Q网络和目标网络
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        # TensorBoard支持
        self.writer = None
        if tensorboard_log_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(tensorboard_log_dir, f"DQN_run_{timestamp}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard日志保存在: {log_dir}")
            print(f"启动TensorBoard命令: tensorboard --logdir {tensorboard_log_dir}")
        
        # 用于记录训练指标
        self.episode_rewards = []
        self.losses = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 计算Q值
        q_values = self.q_net(states).gather(1, actions)
        # 计算目标Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1) 
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 计算损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Training/Loss', dqn_loss.item(), self.training_step)
            self.writer.add_scalar('Training/Q_Values_Mean', q_values.mean().item(), self.training_step)
            self.writer.add_scalar('Training/Q_Targets_Mean', q_targets.mean().item(), self.training_step)
            
            # 记录梯度范数
            total_norm = 0
            for p in self.q_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar('Training/Gradient_Norm', total_norm, self.training_step)
            
        self.training_step += 1

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def log_episode(self, episode, episode_reward, epsilon):
        if self.writer:
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.writer.add_scalar('Episode/Epsilon', epsilon, episode)
            
            # 记录奖励的移动平均
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
            avg_reward = np.mean(self.episode_rewards)
            self.writer.add_scalar('Episode/Average_Reward_100', avg_reward, episode)

    def save_model(self, filepath, episode=None, avg_reward=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.q_net.fc1.in_features,
                'hidden_dim': self.q_net.fc1.out_features,
                'action_dim': self.action_dim,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'target_update': self.target_update,
            },
            'training_info': {
                'episode': episode,
                'avg_reward': avg_reward,
                'training_step': self.training_step,
                'count': self.count,
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
        
        if self.writer:
            self.writer.add_text('Model/Save_Info', 
                f'模型保存: {filepath}\nEpisode: {episode}\nAverage Reward: {avg_reward}', 
                episode if episode else 0)
    
    def load_model(self, filepath, load_optimizer=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            self.training_step = training_info.get('training_step', 0)
            self.count = training_info.get('count', 0)
        else:
            training_info = {}
        
        print(f"模型已加载: {filepath}")
        print(f"超参数: {checkpoint.get('hyperparameters', {})}")
        print(f"训练信息: {training_info}")
        
        return checkpoint
    
    def save_checkpoint(self, checkpoint_dir, episode, avg_reward, best_reward=None):
        """
        保存训练检查点
        
        参数:
            checkpoint_dir: 检查点目录
            episode: 当前训练轮数
            avg_reward: 平均奖励
            best_reward: 最佳奖励
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存当前检查点
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        self.save_model(checkpoint_path, episode, avg_reward)
        
        # 保存最新模型
        latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
        self.save_model(latest_path, episode, avg_reward)
        
        # 如果是最佳模型，额外保存
        if best_reward is not None and avg_reward >= best_reward:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            self.save_model(best_path, episode, avg_reward)
            print(f"新的最佳模型保存: {best_path} (平均奖励: {avg_reward:.2f})")
            
    def close(self):
        """关闭TensorBoard writer"""
        if self.writer:
            self.writer.close()
