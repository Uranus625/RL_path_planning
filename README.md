# 强化学习路径规划项目

这是一个基于DQN/DDQN,PPO算法的路径规划项目，代码已经过模块化重构，方便后续扩展和算法替换。  


## 文件说明
- /dqn 包含dqn和ddqn算法
- /ppo 包含ppo算法
- config.py 参数文件
- env.py 路径规划环境文件，可自行更改其中内容
- rl_utils.py 通用函数与对象
- test_refactored.py 可视化测试脚本
- train.py 训练脚本

## state(14)
- 归一化后的位置 [norm_x, norm_y]
- 归一化后的距离 [norm_distance]
- 归一化后的方向 [norm_dx, norm_dy]
- 上一步动作
- 周围2格的扩展障碍物
- [左1,左2,右1,右2,上1,上2,下1,下2]  
> 在此state下训练，agent具有较强的泛化能力。

## 未来扩展

- [ ] 添加更多RL算法（DDPG,  SAC等）
- [ ] 实现分布式训练
- [ ] 添加更多环境
- [ ] 实现算法性能对比工具
