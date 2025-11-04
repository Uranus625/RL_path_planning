import pygame
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional

class MixedObstacleEnv:
    def __init__(self, 
                 grid_size: Tuple[int, int] = (10, 10),
                 static_obstacle_count: int = 2,
                 moving_obstacle_count: int = 2,
                 static_obstacle_positions: Optional[List[Tuple[int, int]]] = None,
                 moving_obstacle_positions: Optional[List[Tuple[int, int]]] = None,
                 move_prob: float = 0.3,
                 cell_size: int = 20,
                 seed: Optional[int] = None):
        """
        参数:
            grid_size: 网格地图大小 (宽, 高)
            static_obstacle_count: 不可移动障碍物数量
            moving_obstacle_count: 可移动障碍物数量
            static_obstacle_positions: 不可移动障碍物初始位置列表，若为None则随机生成
            moving_obstacle_positions: 可移动障碍物初始位置列表，若为None则随机生成
            move_prob: 每一步可移动障碍物移动的概率
            cell_size: 每个格子的像素大小
            seed: 随机数种子，用于保证实验的可重现性
        """

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        
        self.grid_width, self.grid_height = grid_size
        self.static_obstacle_count = static_obstacle_count
        self.moving_obstacle_count = moving_obstacle_count
        self.move_prob = move_prob  # 可移动障碍物的移动概率
        self.cell_size = cell_size   
        self.action_space = 4
        self.state_space = 3

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.grid_width * cell_size, self.grid_height * cell_size)
        )
        pygame.display.set_caption("路径规划环境 - 混合障碍物类型")

        self.colors = {
            'empty': (255, 255, 255),    # 白色 - 空地
            'agent': (0, 0, 255),       # 蓝色 - 智能体
            'target': (0, 255, 0),      # 绿色 - 目标点
            'static_obstacle': (100, 100, 100),  # 深灰色 - 不可移动障碍物
            'moving_obstacle': (255, 0, 0),      # 红色 - 可移动障碍物
            'grid': (200, 200, 200)     # 浅灰色 - 网格线
        }

        # 初始化智能体位置
        self.agent_pos = self._get_random_position()
        
        # 初始化目标位置（确保与智能体位置不同）
        self.target_pos = self._get_random_position(exclude=[self.agent_pos])
        
        # 存储所有已占用的位置（用于避免重叠）
        occupied_positions = [self.agent_pos, self.target_pos]

        # 初始化不可移动障碍物
        if static_obstacle_positions is not None and len(static_obstacle_positions) >= static_obstacle_count:
            self.static_obstacle_positions = static_obstacle_positions[:static_obstacle_count]
        else:
            self.static_obstacle_positions = []
            for _ in range(static_obstacle_count):
                pos = self._get_random_position(exclude=occupied_positions + self.static_obstacle_positions)
                self.static_obstacle_positions.append(pos)

        # 将不可移动障碍物位置添加到已占用位置
        occupied_positions.extend(self.static_obstacle_positions)
        
        # 初始化可移动障碍物
        if moving_obstacle_positions is not None and len(moving_obstacle_positions) >= moving_obstacle_count:
            self.moving_obstacle_positions = moving_obstacle_positions[:moving_obstacle_count]
        else:
            self.moving_obstacle_positions = []
            for _ in range(moving_obstacle_count):
                pos = self._get_random_position(exclude=occupied_positions + self.moving_obstacle_positions)
                self.moving_obstacle_positions.append(pos)

        # 记录上一步到目标的距离，用于计算距离奖励
        self.previous_distance = self._manhattan_distance(self.agent_pos, self.target_pos)
        
        # 记录上一步的动作（用于状态表示）
        self.last_action = -1  # -1表示没有上一步动作（初始状态）
        
        # 保存初始位置，用于reset_to_initial方法
        self.initial_agent_pos = self.agent_pos
        self.initial_target_pos = self.target_pos
        self.initial_static_obstacle_positions = self.static_obstacle_positions.copy()
        self.initial_moving_obstacle_positions = self.moving_obstacle_positions.copy()

    def _get_random_position(self, exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        if exclude is None:
            exclude = []
            
        while True:
            x = self.np_random.randint(0, self.grid_width)
            y = self.np_random.randint(0, self.grid_height)
            pos = (x, y)
            if pos not in exclude:
                return pos
            
    def _move_moving_obstacles(self) -> None:
        new_moving_positions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        
        # 所有不可移动的位置（用于碰撞检测）
        immovable_positions = [self.agent_pos, self.target_pos] + self.static_obstacle_positions
        
        for obs_pos in self.moving_obstacle_positions:
            # 以一定概率移动障碍物
            if self.np_random.random() < self.move_prob:
                # 随机选择一个方向
                dx, dy = directions[self.np_random.choice(len(directions))]
                new_x = obs_pos[0] + dx
                new_y = obs_pos[1] + dy
                
                # 检查是否在网格内且不与其他物体重叠
                if (0 <= new_x < self.grid_width and 
                    0 <= new_y < self.grid_height and 
                    (new_x, new_y) not in immovable_positions and 
                    (new_x, new_y) not in new_moving_positions):
                    new_moving_positions.append((new_x, new_y))
                else:
                    new_moving_positions.append(obs_pos)
            else:
                new_moving_positions.append(obs_pos)
        
        self.moving_obstacle_positions = new_moving_positions

    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        return (pos in self.static_obstacle_positions or 
                pos in self.moving_obstacle_positions)
    
    def _is_out_of_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_surrounding_obstacles(self, pos: Tuple[int, int]) -> List[int]:
        """
        获取agent周围四个方向是否有障碍物的信息
        
        参数:
            pos: 要检查的位置
            
        返回:
            长度为4的列表，表示[左、右、上、下]四个方向是否有障碍物
            0表示可通行，1表示有障碍物或出界
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        obstacles = []
        
        for dx, dy in directions:
            next_pos = (pos[0] + dx, pos[1] + dy)
            # 检查是否有障碍物或出界
            if self._is_out_of_bounds(next_pos) or self._is_collision(next_pos):
                obstacles.append(1)
            else:
                obstacles.append(0)
        
        return obstacles
    
    def _get_extended_obstacles(self, pos: Tuple[int, int], radius: int = 2) -> List[int]:
        """
        获取agent周围扩展范围内的障碍物信息（四个方向的射线检测）
        
        参数:
            pos: agent当前位置
            radius: 感知半径（默认2格）
            
        返回:
            列表，包含四个方向上radius格范围内的障碍物信息
            每个方向返回radius个值：[左1,左2,..., 右1,右2,..., 上1,上2,..., 下1,下2,...]
            例如radius=2时返回8个值：[左1,左2, 右1,右2, 上1,上2, 下1,下2]
            0表示可通行，1表示有障碍物或出界
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        obstacles = []
        
        for dx, dy in directions:
            # 对每个方向，检测从1到radius格的距离
            for distance in range(1, radius + 1):
                check_pos = (pos[0] + dx * distance, pos[1] + dy * distance)
                if self._is_out_of_bounds(check_pos) or self._is_collision(check_pos):
                    obstacles.append(1)
                else:
                    obstacles.append(0)
        
        return obstacles
    
    def _get_square_obstacles(self, pos: Tuple[int, int], radius: int = 1) -> List[int]:
        """
        获取agent周围方形区域内的障碍物分布（真正的NxN区域）
        
        参数:
            pos: agent当前位置
            radius: 方形区域半径（radius=1表示3x3区域，radius=2表示5x5区域）
            
        返回:
            列表，包含周围方形区域的障碍物信息（不包括agent自己所在位置）
            按行扫描顺序返回：从上到下，从左到右
            
        示例：radius=1时（3x3区域，返回8个值）
            □ □ □
            □ A □  → [左上, 上, 右上, 左, 右, 左下, 下, 右下]
            □ □ □
            
        示例：radius=2时（5x5区域，返回24个值）
            □ □ □ □ □
            □ □ □ □ □
            □ □ A □ □
            □ □ □ □ □
            □ □ □ □ □
        """
        obstacles = []
        
        # 扫描以pos为中心的(2*radius+1) x (2*radius+1)方形区域
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # 跳过agent自己的位置
                if dx == 0 and dy == 0:
                    continue
                
                check_pos = (pos[0] + dx, pos[1] + dy)
                if self._is_out_of_bounds(check_pos) or self._is_collision(check_pos):
                    obstacles.append(1)
                else:
                    obstacles.append(0)
        
        return obstacles

    def _get_state(self) -> Dict:
        # 计算到目标的相对位置
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        
        # 获取周围障碍物信息（近距离：1格）
        surrounding_obstacles = self._get_surrounding_obstacles(self.agent_pos)
        
        # 获取扩展范围障碍物信息（远距离：2格范围）
        extended_obstacles = self._get_extended_obstacles(self.agent_pos, radius=2)
        
        # 归一化坐标（0-1范围）
        normalized_agent_x = self.agent_pos[0] / self.grid_width
        normalized_agent_y = self.agent_pos[1] / self.grid_height
        
        # 归一化目标方向（-1到1范围）
        max_distance = self.grid_width + self.grid_height  # 地图对角线的曼哈顿距离
        normalized_dx = dx / self.grid_width
        normalized_dy = dy / self.grid_height
        
        return { 
            'normalized_agent_pos': (normalized_agent_x, normalized_agent_y),
            'normalized_distance': self._manhattan_distance(self.agent_pos, self.target_pos) / max_distance,
            'normalized_target_direction': (normalized_dx, normalized_dy),
            # 'surrounding_obstacles': surrounding_obstacles,  # 周围1格的障碍物 [左,右,上,下] 共4个
            'extended_obstacles': extended_obstacles,  # 周围2格的障碍物 [左1,左2,右1,右2,上1,上2,下1,下2] 共8个
            'last_action': self.last_action  # 上一步的动作
        }

    def reset(self, new_target: Optional[Tuple[int, int]] = None) -> Dict:
        """
        重置环境
        
        参数:
            new_target: 新的目标位置，若为None则随机生成
        
        返回:
            初始状态
        """
        # 重置智能体位置（避开所有障碍物）
        all_obstacles = self.static_obstacle_positions + self.moving_obstacle_positions
        self.agent_pos = self._get_random_position(exclude=all_obstacles)
        
        # 重置目标位置
        if new_target is not None and not self._is_collision(new_target) and not self._is_out_of_bounds(new_target):
            self.target_pos = new_target
        else:
            exclude_positions = [self.agent_pos] + all_obstacles
            self.target_pos = self._get_random_position(exclude=exclude_positions) 
        
        # 重置距离记录
        self.previous_distance = self._manhattan_distance(self.agent_pos, self.target_pos)
        
        # 重置上一步动作
        self.last_action = -1
        
        # 返回初始状态
        return self._get_state()

    def reset_to_initial(self) -> Dict:
        """
        重置环境到初始实例化时的位置
        保持agent_pos、target_pos和所有障碍物的初始位置
        
        返回:
            初始状态
        """
        # 恢复到初始位置
        self.agent_pos = self.initial_agent_pos
        self.target_pos = self.initial_target_pos
        self.static_obstacle_positions = self.initial_static_obstacle_positions.copy()
        self.moving_obstacle_positions = self.initial_moving_obstacle_positions.copy()
        
        # 重置距离记录
        self.previous_distance = self._manhattan_distance(self.agent_pos, self.target_pos)
        
        # 重置上一步动作
        self.last_action = -1
        
        # 返回初始状态
        return self._get_state()

    def reset_agent_only(self, new_agent_pos=None) -> Dict:
        """
        只重置智能体位置，保持目标位置和障碍物位置不变
        
        参数:
            new_agent_pos: 新的智能体位置，如果为None则随机生成一个有效位置
        
        返回:
            新状态
        """
        # 保持目标位置和障碍物位置不变
        self.target_pos = self.initial_target_pos
        self.static_obstacle_positions = self.initial_static_obstacle_positions.copy()
        self.moving_obstacle_positions = self.initial_moving_obstacle_positions.copy()
        
        # 设置智能体位置
        if new_agent_pos is not None:
            # 检查新位置是否有效
            if (not self._is_out_of_bounds(new_agent_pos) and 
                not self._is_collision(new_agent_pos) and 
                new_agent_pos != self.target_pos):
                self.agent_pos = new_agent_pos
            else:
                print(f"警告: 指定位置 {new_agent_pos} 无效，使用随机位置")
                self.agent_pos = self._get_random_agent_position()
        else:
            # 随机生成一个有效的智能体位置
            self.agent_pos = self._get_random_agent_position()
        
        # 重置距离记录
        self.previous_distance = self._manhattan_distance(self.agent_pos, self.target_pos)
        
        # 重置上一步动作
        self.last_action = -1
        
        # 返回新状态
        return self._get_state()

    def _get_random_agent_position(self):
        """获取一个随机的有效智能体位置"""
        all_obstacles = self.static_obstacle_positions + self.moving_obstacle_positions
        occupied_positions = [self.target_pos] + all_obstacles
        return self._get_random_position(exclude=occupied_positions)

### TODO：完成step方法（奖励函数）
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        dx, dy = directions[action]

        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        new_pos = (new_x, new_y)

        collision = self._is_collision(new_pos)
        out_of_bounds = self._is_out_of_bounds(new_pos)

        reward = -1.0  # 每运动一步存在基础惩罚

        if not collision and not out_of_bounds:
            old_pos = self.agent_pos
            self.agent_pos = new_pos
            
            # 计算距离奖励
            current_distance = self._manhattan_distance(self.agent_pos, self.target_pos)
            distance_reward = self.previous_distance - current_distance  # 距离减少为正奖励
            reward += distance_reward * 5.0  # 降低倍数，避免过度主导
            
            # 强化惩罚反向移动（回到刚离开的位置）
            opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}  # 左<->右, 上<->下
            if self.last_action != -1 and action == opposite_actions[self.last_action]:
                reward -= 11.0  # 加大惩罚，使反复横跳总体为负
            
            # 更新距离记录
            self.previous_distance = current_distance
            
            # 记录当前动作（在成功移动后）
            self.last_action = action
        else:
            reward -= 25.0  # 碰撞或出界的惩罚
            # 碰撞时不更新last_action，因为动作失败了

        # 检查是否到达目标
        done = False
        if self.agent_pos == self.target_pos:
            reward += 100.0  # 到达目标的奖励
            done = True

        self._move_moving_obstacles()

        state = self._get_state()

        info = {
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'collision_type': 'static' if collision and new_pos in self.static_obstacle_positions else 
                             'moving' if collision else None,
            'distance_to_target': self._manhattan_distance(self.agent_pos, self.target_pos)
        }
        return state, reward, done, info

    def render(self, delay: float = 0.1) -> None:
        """渲染环境"""
        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 填充背景
        self.screen.fill(self.colors['empty'])
        
        # 绘制网格线
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)
        
        # 绘制不可移动障碍物
        for (x, y) in self.static_obstacle_positions:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                              self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['static_obstacle'], rect)
        
        # 绘制可移动障碍物
        for (x, y) in self.moving_obstacle_positions:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                              self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['moving_obstacle'], rect)
        
        # 绘制目标
        tx, ty = self.target_pos
        rect = pygame.Rect(tx * self.cell_size, ty * self.cell_size, 
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['target'], rect)
        
        # 绘制智能体
        ax, ay = self.agent_pos
        rect = pygame.Rect(ax * self.cell_size, ay * self.cell_size, 
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['agent'], rect)
        
        # 更新显示
        pygame.display.flip()
        
        # 延迟，控制显示速度
        time.sleep(delay)
    
    def close(self) -> None:
        """关闭环境"""
        pygame.quit()


if __name__ == "__main__":
    # 简单测试环境
    env = MixedObstacleEnv(
        grid_size=(40, 40),
        static_obstacle_count=250,
        moving_obstacle_count=100,
        move_prob=0.2,
        seed=42
    )

    state = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 50:
        action = np.random.randint(0, env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"步骤 {step_count}: 动作={action}, 奖励={reward:.2f}, 状态={next_state}, 信息={info}")
        env.render(delay=0.2)
        step_count += 1

    env.close()

