import torch

def get_config():
    """
    返回:
        配置字典
    """
    config = {
        # 环境参数
        'env': {
            'grid_size': (40, 40),
            'static_obstacle_count': 250,
            'moving_obstacle_count': 100,
            'move_prob': 0.6,
            'seed': 44
        },
        
        # 网络参数
        'network': {
            'hidden_dim': 128,
        },
        
        # 训练参数
        'training': {
            'num_episodes': 20000,
            'learning_rate': 2e-3,
            'gamma': 0.95,
            'epsilon_start': 0.95,
            'epsilon_end': 0.2,
            'epsilon_decay': 0.9999,
            'target_update': 10,
            'batch_size': 64,
            'epochs': 10,
            'actor_lr': 1e-3,
            'critic_lr': 1e-2,
            'lmbda': 0.95,
            'eps': 0.2,
        },
        
        # 经验回放参数
        'replay_buffer': {
            'buffer_size': 10000,
            'minimal_size': 2000,
        },
        
        # 保存和日志参数
        'logging': {
            'tensorboard_log_dir': './tensorboard_log',
            'checkpoint_dir': './saved_models/checkpoints',
            'dqn_model_path': './saved_models/dqn_model.pt',
            'ppo_model_path': './saved_models/ppo_model.pt',
            'save_interval': 500,  # 每多少个episode保存一次检查点
        },
        
        # 设备参数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 随机种子
        'seed': 42,
    }
    
    return config


def update_config(config, updates):
    import copy
    new_config = copy.deepcopy(config)
    
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = recursive_update(d[k], v)
            else:
                d[k] = v
        return d
    
    return recursive_update(new_config, updates)
