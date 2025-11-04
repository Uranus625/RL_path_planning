import numpy as np


class DQNTrainer:
    """DQN训练器"""
    
    def __init__(self, env, agent, replay_buffer, config):
        """
        初始化训练器
        
        参数:
            env: 环境实例
            agent: DQN智能体实例
            replay_buffer: 经验回放池实例
            config: 配置字典
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = config
        
        # 训练统计
        self.return_list = []
        self.best_avg_reward = -float('inf')
    
    def train(self, num_episodes, state_to_vector_fn, verbose=True):
        """
        执行训练
        
        参数:
            num_episodes: 训练的episode数
            state_to_vector_fn: 状态转向量的函数
            verbose: 是否打印训练信息
            
        返回:
            训练过程中的回报列表
        """
        epsilon = self.config['epsilon_start']
        minimal_size = self.config['minimal_size']
        batch_size = self.config['batch_size']
        epsilon_decay = self.config['epsilon_decay']
        epsilon_end = self.config['epsilon_end']
        
        if verbose:
            print("开始训练...")
        
        for episode in range(num_episodes):
            episode_return = 0
            state_dict = self.env.reset()
            state_vector = state_to_vector_fn(state_dict)
            done = False
            
            # 执行一个episode
            while not done:
                action = self.agent.take_action(state_vector)
                next_state_dict, reward, done, info = self.env.step(action)
                next_state_vector = state_to_vector_fn(next_state_dict)
                
                # 存储经验
                self.replay_buffer.add(state_vector, action, reward, next_state_vector, done)
                state_vector = next_state_vector
                episode_return += reward
                
                # 训练智能体
                if self.replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    self.agent.update(transition_dict)
            
            # 记录回报
            self.return_list.append(episode_return)
            
            # 记录到TensorBoard
            self.agent.log_episode(episode, episode_return, epsilon)
            
            # epsilon衰减
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            self.agent.epsilon = epsilon
            
            # 计算平均奖励
            if len(self.return_list) >= 100:
                avg_reward = np.mean(self.return_list[-100:])
            else:
                avg_reward = np.mean(self.return_list)
            
            # 打印进度
            if verbose and (episode + 1) % 50 == 0:
                avg_return = np.mean(self.return_list[-50:])
                print(f'Episode [{episode+1:4d}/{num_episodes}] - '
                      f'Avg Return: {avg_return:7.2f}, Epsilon: {epsilon:.3f}')
            
            # # 保存检查点
            # if (episode + 1) % self.config.get('save_interval', 500) == 0:
            #     checkpoint_dir = self.config.get('checkpoint_dir', './saved_models/checkpoints')
            #     self.agent.save_checkpoint(checkpoint_dir, episode + 1, avg_reward, self.best_avg_reward)
            #     if avg_reward > self.best_avg_reward:
            #         self.best_avg_reward = avg_reward
        
        return self.return_list
    
    def save_final_model(self):
        """保存最终模型"""
        final_model_path = self.config.get('final_model_path', './saved_models/final_model.pt')
        final_avg_reward = (np.mean(self.return_list[-100:]) 
                           if len(self.return_list) >= 100 
                           else np.mean(self.return_list))
        self.agent.save_model(final_model_path, len(self.return_list), final_avg_reward)
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最终平均奖励 (最后100轮): {final_avg_reward:.2f}")
        print("\n保存的模型文件:")
        print(f"  - 最终模型: {final_model_path}")
        print(f"  - 检查点模型: {self.config.get('checkpoint_dir', './saved_models/checkpoints/')}")
        print(f"\nTensorBoard日志: tensorboard --logdir {self.config.get('tensorboard_log_dir', './dqn')}")
