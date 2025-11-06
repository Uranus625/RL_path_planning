"""
训练脚本
使用DQN算法训练智能体
"""

import torch
import numpy as np
import env
from dqn import DQNAgent, DQNTrainer
from ppo import PPOAgent, PPOTrainer
from rl_utils import state_to_vector, ReplayBuffer
from config import get_config


def main():
    """主训练函数"""
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 创建环境
    environment = env.MixedObstacleEnv(**config['env'])
    
    # 获取状态和动作维度
    sample_state = environment.reset()
    state_vector = state_to_vector(sample_state)
    state_dim = len(state_vector)
    action_dim = environment.action_space
    
    print(f"使用设备: {config['device']}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 创建设备
    device = torch.device(config['device'])
    
    # # 创建经验回放池
    # replay_buffer = ReplayBuffer(config['replay_buffer']['buffer_size'])
    
    # # 创建DQN智能体
    # agent = DQNAgent(
    #     state_dim=state_dim,
    #     hidden_dim=config['network']['hidden_dim'],
    #     action_dim=action_dim,
    #     learning_rate=config['training']['learning_rate'],
    #     gamma=config['training']['gamma'],
    #     epsilon=config['training']['epsilon_start'],
    #     target_update=config['training']['target_update'],
    #     device=device,
    #     tensorboard_log_dir=config['logging']['tensorboard_log_dir'],
    #     dqn_type='DoubleDQN'
    # )
    
    # # 合并配置以便传递给训练器
    # trainer_config = {
    #     'epsilon_start': config['training']['epsilon_start'],
    #     'epsilon_end': config['training']['epsilon_end'],
    #     'epsilon_decay': config['training']['epsilon_decay'],
    #     'minimal_size': config['replay_buffer']['minimal_size'],
    #     'batch_size': config['training']['batch_size'],
    #     'checkpoint_dir': config['logging']['checkpoint_dir'],
    #     'dqn_model_path': config['logging']['dqn_model_path'],
    #     'save_interval': config['logging']['save_interval'],
    #     'tensorboard_log_dir': config['logging']['tensorboard_log_dir'],
    # }
    
    # # 创建训练器
    # trainer = DQNTrainer(environment, agent, replay_buffer, trainer_config)
    
    agent = PPOAgent(
        state_dim=state_dim,
        hidden_dim=config['network']['hidden_dim'],
        action_dim=action_dim,
        actor_lr=config['training']['actor_lr'],
        critic_lr=config['training']['critic_lr'],
        lmbda=config['training']['lmbda'],
        eps=config['training']['eps'],
        epochs=config['training']['epochs'],
        gamma=config['training']['gamma'],
        device=device,
        tensorboard_log_dir=config['logging']['tensorboard_log_dir'],
    )

    trainer = PPOTrainer(environment, agent)

    # 开始训练
    return_list = trainer.train(
        num_episodes=config['training']['num_episodes'],
        state_to_vector_fn=state_to_vector,
        verbose=True
    )
    
    # 保存最终模型
    trainer.save_final_model()
    
    # 关闭环境和agent
    environment.close()
    agent.close()
    
    print("\n模型加载示例:")
    print(f"  agent.load_model('{config['logging']['dqn_model_path']}')")


if __name__ == "__main__":
    main()
