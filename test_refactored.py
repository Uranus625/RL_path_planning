import torch
import numpy as np
import time
import env
from dqn import DQNAgent
from ppo import PPOAgent
from rl_utils import state_to_vector
from config import get_config


def test_trained_model(model_path, num_test_episodes=5, render=True, render_delay=0.3):
    """
    测试训练好的DQN模型
    
    参数:
        model_path: 模型文件路径
        num_test_episodes: 测试的episode数
        render: 是否渲染环境
        render_delay: 渲染延迟时间
    """
    # 获取配置
    config = get_config()
    
    # 创建环境
    environment = env.MixedObstacleEnv(**config['env'])
    
    # 获取状态和动作维度
    sample_state = environment.reset()
    state_vector = state_to_vector(sample_state)
    state_dim = len(state_vector)
    action_dim = environment.action_space
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"网格大小: {environment.grid_width} x {environment.grid_height}")
    print(f"静态障碍物数量: {environment.static_obstacle_count}")
    print(f"移动障碍物数量: {environment.moving_obstacle_count}")
    
    # 创建智能体（测试时epsilon=0，完全利用）
    device = torch.device(config['device'])
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
    # agent = DQNAgent(
    #     state_dim=state_dim,
    #     hidden_dim=config['network']['hidden_dim'],
    #     action_dim=action_dim,
    #     learning_rate=config['training']['learning_rate'],
    #     gamma=config['training']['gamma'],
    #     epsilon=0.0,  # 测试时不探索
    #     target_update=config['training']['target_update'],
    #     device=device
    # )
    
    # 加载模型
    try:
        agent.load_model(model_path, load_optimizer=False)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        environment.close()
        return
    
    print(f"\n开始测试训练好的模型...")
    print("=" * 50)
    
    # 测试统计
    total_rewards = []
    total_steps = []
    success_count = 0
    
    # 动作名称映射
    action_names = ['左', '右', '上', '下']
    
    for episode in range(num_test_episodes):
        print(f"\n测试回合 {episode + 1}/{num_test_episodes}")
        
        state_dict = environment.reset()
        state_vector = state_to_vector(state_dict)
        
        print(f"初始位置 - 智能体: {environment.agent_pos}, 目标: {environment.target_pos}")
        
        episode_reward = 0
        step_count = 0
        done = False
        max_steps = 100  # 防止无限循环
        
        # 显示初始环境
        if render:
            environment.render(delay=0.5)
        
        while not done and step_count < max_steps:
            # 使用训练好的策略选择动作
            action = agent.take_action(state_vector)
            
            # 执行动作
            next_state_dict, reward, done, info = environment.step(action)
            next_state_vector = state_to_vector(next_state_dict)
            
            episode_reward += reward
            step_count += 1
            
            print(f"步骤 {step_count}: 动作={action_names[action]}, "
                  f"奖励={reward:.2f}, 位置={environment.agent_pos}")
            
            # 显示环境
            if render:
                environment.render(delay=render_delay)
            
            # 检查是否到达目标
            if done:
                if environment.agent_pos == environment.target_pos:
                    print(f"✅ 成功到达目标！用了 {step_count} 步")
                    success_count += 1
                else:
                    print(f"❌ 回合结束，未到达目标")
                break
            
            state_vector = next_state_vector
        
        if step_count >= max_steps:
            print(f"❌ 达到最大步数限制 ({max_steps} 步)")
        
        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        
        print(f"回合 {episode + 1} 结果: 总奖励={episode_reward:.2f}, 步数={step_count}")
        print("-" * 30)
        
        # 等待一下再进行下一回合
        if render:
            time.sleep(1)
    
    # 测试结果统计
    print("\n" + "=" * 50)
    print("测试结果统计:")
    print(f"成功率: {success_count}/{num_test_episodes} "
          f"({success_count/num_test_episodes*100:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
    print(f"最佳回合奖励: {max(total_rewards):.2f}")
    print(f"最少步数: {min(total_steps)}")
    
    environment.close()


if __name__ == "__main__":
    model_path = "./saved_models/ppo_model.pt"
    test_trained_model(model_path, num_test_episodes=15, render=True)
