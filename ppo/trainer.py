import numpy as np
import time

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
        # 训练统计
        self.return_list = []
    
    def train(self, num_episodes, state_to_vector_fn, verbose=True):
        if verbose:
            print("开始PPO训练...")
            print(f"目标episode数: {num_episodes}")
            print("="*60)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = self.env.reset()
            state_vector = state_to_vector_fn(state)
            done = False
            steps = 0
            
            # 收集一个完整episode的数据
            while not done:
                action = self.agent.take_action(state_vector)
                next_state, reward, done, info = self.env.step(action)
                next_state_vector = state_to_vector_fn(next_state)
                transition_dict['states'].append(state_vector)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state_vector)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state_vector = next_state_vector
                episode_return += reward
                steps += 1
            
            # episode结束后,用整个episode的数据更新一次
            self.agent.update(transition_dict)
            
            self.return_list.append(episode_return)

            self.agent.log_episode(episode, episode_return)

            # 更频繁的进度显示
            if verbose and (episode + 1) % 50 == 0:
                # 安全地计算平均值，避免空切片警告
                avg_return = np.mean(self.return_list[-50:]) if len(self.return_list) >= 50 else np.mean(self.return_list) if self.return_list else 0.0
                print(f'Episode [{episode+1:4d}/{num_episodes}] - '
                      f'Avg Return: {avg_return:7.2f}')
        
        return self.return_list
    
    def save_final_model(self):
        """保存最终模型"""
        final_model_path = './saved_models/ppo_model.pt'
        # 安全地计算最终平均奖励
        if len(self.return_list) >= 100:
            final_avg_reward = np.mean(self.return_list[-100:])
        elif self.return_list:
            final_avg_reward = np.mean(self.return_list)
        else:
            final_avg_reward = 0.0
        self.agent.save_model(final_model_path, len(self.return_list), final_avg_reward)
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最终平均奖励 (最后100轮): {final_avg_reward:.2f}")
        print("\n保存的模型文件:")
        print(f"  - 最终模型: {final_model_path}")
        print(f"\nTensorBoard日志: tensorboard --logdir {'./tensorboard_log'}")