import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import rl_utils
from rl_utils import PolicyNet, ValueNet

class PPOAgent:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, tensorboard_log_dir=None):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 初始化网络权重
        self._init_weights(self.actor)
        self._init_weights(self.critic)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma  # 折扣因子，用于控制未来回报的权重。
        self.lmbda = lmbda  # GAE 的衰减系数，控制长期和短期偏差的平衡。
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.tensorboard_log_dir = tensorboard_log_dir
        
        # 用于记录训练指标
        self.training_step = 0
        self.episode_rewards = []  # 用于计算移动平均

        self.writer = None
        self.log_dir = None
        if tensorboard_log_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(tensorboard_log_dir, f"PPO_run_{timestamp}")
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard日志保存在: {self.log_dir}")
            print(f"启动TensorBoard命令: tensorboard --logdir {tensorboard_log_dir}")
    
    def _init_weights(self, model):
        """初始化网络权重"""
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        
        # 检查是否有NaN
        if torch.isnan(probs).any():
            # print(f"警告: 网络输出包含NaN！state: {state}")
            # print(f"probs: {probs}")
            # 使用均匀分布作为fallback
            probs = torch.ones_like(probs) / probs.shape[1]
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def log_episode(self, episode, episode_reward):
        if self.writer:
            self.writer.add_scalar('Episode/Reward', episode_reward, episode)
            
            # 记录奖励的移动平均
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
            # 安全地计算平均值
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards)
                self.writer.add_scalar('Episode/Average_Reward_100', avg_reward, episode)
    

    def save_model(self, filepath, episode=None, avg_reward=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'actor_dict': self.actor.state_dict(),
            'critic_dict': self.critic.state_dict(),
            'actor_optimizer_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim,
                'action_dim': self.action_dim,
                'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
                'lmbda': self.lmbda,
                'gamma': self.gamma,
                'eps': self.eps,
                'epochs': self.epochs,
            },
            'training_info': {
                'episode': episode,
                'avg_reward': avg_reward,
                'training_step': self.training_step,
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

        self.actor.load_state_dict(checkpoint['actor_dict'])
        self.critic.load_state_dict(checkpoint['critic_dict'])

        if load_optimizer and 'actor_optimizer_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_dict'])
        if load_optimizer and 'critic_optimizer_dict' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_dict'])

        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            self.training_step = training_info.get('training_step', 0)
        else:
            training_info = {}
        
        print(f"模型已加载: {filepath}")
        print(f"超参数: {checkpoint.get('hyperparameters', {})}")
        print(f"训练信息: {training_info}")
        
        return checkpoint
    
    def close(self):
        """关闭TensorBoard writer"""
        if self.writer:
            self.writer.close()