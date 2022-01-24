from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.utils.reward_constructor import Reward
from operator import add
import numpy as np
import os

class FollowTheLeaderEnv(MiniGridEnv):
    """
    Single-room square grid environment with moving obstacles
    """
    
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        toggle = 3
    
    def __init__(
            self,
            size=20,
            agent_start_pos=(5,1),
            agent_start_dir=0,
            leader_start_pos=(5,3),
            min_distance = 1,
            max_distance = 4,
            max_dev = 1,
            warm_start = 5,
            movement_strategy = "forward", 
            random_step = 3,
            max_step_size = None,
            reward_config = None,
            n_obstacles = 0
            #available_actions = ["forward", "turn_left", "turn_right", "left", "right", "message_stop", "speed_up", "speed_down"] # пока не используется
            
    ):
        '''Класс для создания среды следования за лидером.
        Входные параметры:
        size -- int:
            Размер стороны поля (по умолчанию: 20);
            
        agent_start_pos -- tuple(int,int):
            Стартовые координаты агента;
            
        agent_start_dir -- int:
            Стартовое направление агента;
        
        leader_start_pos -- tuple(int,int):
            Стартовое направление ведущего;
            
        min_distance -- int: 
            Минимальная дистанция, которую должен держать агент от ведущего;
        
        max_distance -- int: 
            Максимальная дистанция, дальше которой агент не должен отставать;
        
        max_dev -- int: 
            Максимальное отклонение от маршрута (в клетках);
            
        warm_start -- int: 
            Число шагов среды, в течение которых агент не получает штраф за движение вне маршрута (чтобы он успел встать на маршрут);
            
        movement_strategy -- str, List(str) or "random":
            название файла из папки movement_strategies без расширения или список названий (в таком случае при каждом запуске среды маршрут выбирается случайно) 
            или "random" -- движение в случайном направлении.
        
        random_step -- int или None:
            иcпользуется только если movement_strategy = random, показывает, сколько шагов будет совершено в случайном направлении перед сменой.
            Если None, определяется случайно в диапазоне от 2 до 10.
            
        max_step_size -- int or None:
            число шагов среды до конца симуляции. Если None, определяется по числу действий ведущего.
            
        reward_config -- str or None:
            путь до json-файла, который описывает награду в соответствии с dataclass Reward из gym_minigrid.utils.reward_constructor. 
            Если None, используется значение по умолчанию;
            
        see_through_walls -- bool:
            Способен ли агент видеть сквозь стены (True для ускорения работы и обучения)
            
        seed -- int:
            random seed;
            
        agent_view_size -- int:
            Размер стороны поля зрения агента;
        
        n_obstacles -- число случайно генерируемых препятствий на поле.
        '''
        
        self.agent_start_pos = np.array(agent_start_pos,dtype=int)
        self.agent_start_dir = agent_start_dir
        
        self.leader_start_pos = np.array(leader_start_pos,dtype=int)
        
        self.n_obstacles = n_obstacles
        
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_dev = max_dev
        self.movement_strategy = movement_strategy
        
        if movement_strategy == "random":
            self.random_step = random_step
        
        self.warm_start = warm_start
        self.accumulated_reward = 0
        
        
        self.leader_movement_strategy = self._determine_movement_strategy()
        
        if max_step_size is None:
            max_step_size = len(self.leader_movement_strategy)
        
        self.simulation_nb = 0
        
        # TODO: перетащить оттуда нужное и убрать этот дурацкий вызов
        super().__init__(
            grid_size=size,
            max_steps=max_step_size,#4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
                
        self.action_space = spaces.Discrete(self.actions.toggle + 1)
        
        if reward_config:
            self.reward_config = Reward.from_json(reward_config)
        else:
            self.reward_config = Reward()
            
        self.reward_range = (-100, 100)
        
        self.reset()
        
        
    
    
    def reset(self):
        super().reset()
        
        print("===Запуск симуляции номер {}===".format(self.simulation_nb))
        print()
        self.leader_step = 0
        self.leader_trace = list()
        
        if self.simulation_nb > 0:
            self.leader_movement_strategy = self._determine_movement_strategy()
        
        
        self.stop_signal = False
        self.crash = False
        
        self.accumulated_reward = 0
        self.simulation_nb += 1

    
    def _determine_movement_strategy(self):
        if isinstance(self.movement_strategy,list):
            cur_strategy_name = strategy_name(self._rand_int(0,len(self.movement_strategy)))
        else:
            cur_strategy_name = self.movement_strategy
            
            
        if cur_strategy_name == "random":
            leader_movement_strategy = self.random_strategy_generate(cur_strategy_name)
        else:
            leader_movement_strategy = self.movement_strategy_generate(cur_strategy_name)
        
        return leader_movement_strategy
    
    
    @staticmethod
    def _coords_diff(first_tuple, second_tuple):
        # Minkowski Distance
        return sum([abs(i-j) for i,j in zip(first_tuple, second_tuple)])
    
    @staticmethod
    def distance_on_trace():
        raise NotImplementedError()
    
    @staticmethod
    def add_skipped_points():
        raise NotImplementedError()

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = np.array(self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Создание ведущего
        self.leader = Ball()
        self.put_obj(self.leader, *self.leader_start_pos)

        self.mission = "Следуй за лидером, минимальная дистанция = {0}, максимальная = {1},\n отклонение от маршрута = {2},шагов без штрафа = {3}.".format(self.min_distance, self.max_distance, self.max_dev, self.warm_start)
        
    
    
    def step(self, action):
        reward = 0
        
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
#         front_cell = self.grid.get(*self.front_pos)
#         not_clear = front_cell and front_cell.type != 'floor'

        #Здесь определяем движение Ведущего
        old_pos = self.leader.cur_pos
        self.leader_trace.append(old_pos)
        
        # обработка стоп-сигнала и движение ведущего
        if self.stop_signal:
            leader_movement = np.zeros(2,dtype=int)
#             reward -= 1
            print("Ведущий стоит по просьбе агента.")
        else:
            if self.leader_step >= len(self.leader_movement_strategy):
                leader_movement = np.zeros(2,dtype=int)
                print("Лидер прибыл в точку назначения.", self.leader_step, self.leader_movement_strategy)
            else:
                leader_movement = self.leader_movement_strategy[self.leader_step]
            
            if self.grid.get(*self.leader.cur_pos+leader_movement) not in {None, "floor"}:
                leader_movement = np.zeros(2,dtype=int)
            
            self.leader_step+=1
#             reward += 1
        
        self.put_obj(self.leader,*(old_pos+leader_movement))
        
        
        
        size_of_reward_box = self.max_distance - self.min_distance
        unique_points_iter = 0
        
        
        self.is_on_trace = False # находится ли агент на маршруте;
        self.is_in_box = False # находится ли агент в "коробке";
        trace_diff = 0
        
        
        if self.leader.cur_pos != old_pos:
            # просто np_unique нельзя, потому что ведущий в теории может вернуться в ту же точку.    
            for cur_trace_point_id in range(len(self.leader_trace)):# zip(reversed(self.leader_trace[:-1]), reversed(self.leader_trace[1:])):

                reverse_id = len(self.leader_trace)-cur_trace_point_id-1

                cur_leader_trace_point = self.leader_trace[reverse_id]

                if reverse_id != 0:
                    prev_leader_trace_point = self.leader_trace[reverse_id-1]
                else:
                     prev_leader_trace_point = cur_leader_trace_point


                self.put_obj(Floor("yellow"), *cur_leader_trace_point)

                if np.array_equal(self.agent_pos, cur_leader_trace_point):
                        self.is_on_trace = True


                if np.any(cur_leader_trace_point!=prev_leader_trace_point):
                    unique_points_iter += 1

                if (unique_points_iter > self.min_distance) and (unique_points_iter <= self.max_distance):
                    self.put_obj(Floor("red"), *cur_leader_trace_point)
    #                 self.cur_bounding_box.append(cur_leader_trace_point)

                    trace_diff = sum(abs(self.agent_pos - cur_leader_trace_point))
                    if (trace_diff==0) or (trace_diff <= self.max_dev and not self.is_on_trace):
                        self.is_in_box = True

        
        # Обработка действий агента
        done = self._agent_action_processing(action)
        
        reward = self._reward_computation()
            
        self.accumulated_reward += reward
        
        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        print()
        print("Аккумулированная награда: {}".format(self.accumulated_reward))
        print()
        
        print("step", self.step_count, "action", )
        
        
        return obs, reward, done, {}# info

    
    
    def movement_strategy_generate(self, strategy_name):
            
        print("выбранная стратегия движения: ", strategy_name)
        
        with open("movement_strategies/{}.txt".format(strategy_name), "r") as strat_file:
            strategy_commands = strat_file.readlines()
        
        list_commands_by_step = list()
        for cur_command in strategy_commands:
            t_str = cur_command.strip("\n")
            step_nb, first_coord, sec_coord = cur_command.split(";")
            
            for i in range(1,int(step_nb)):
                list_commands_by_step.append(np.array([int(first_coord),int(sec_coord)], dtype=int))
        
        return list_commands_by_step
        
    
    def random_strategy_generate(self):
        
        list_commands_by_step = list()
        
        for cur_command_nb in range(int(self.max_step_size/self.random_step)):
            cur_command = np.array((self._randint(-1,1), self._randint(-1,1)))
            for i in range(self.random_step):
                list_commands_by_step.append(cur_command.copy())
        
        return list_commands_by_step
    
    
    def _agent_action_processing(self, action):
        done = False
        
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'floor'

        
        #Rotate_left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        
        
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
                
            # If the agent tried to walk over an obstacle or wall
            
            
            elif not_clear:
                print("Авария!")
                self.crash = True
                done = True
                
                
                
#             Пока лавы нет, это не нужно
#             if fwd_cell != None and fwd_cell.type == 'lava':
#                 done = True

        # Stop_signal
        elif action == self.actions.toggle:
            self.stop_signal = not self.stop_signal
            print("стоп-сигнал -- {}".format(self.stop_signal))
            
        return done
    
    
            # If the agent tried to walk over an obstacle or wall
#         if action == self.actions.forward and not_clear:
#             print("Авария!")
#             self.crash = True
#             done = True

    
    
    
    def _reward_computation(self):
        # Скорее всего, это можно сделать красивее
        res_reward = 0
        
        if self.stop_signal:
            res_reward += self.reward_config.leader_stop_penalty
            print("Лидер стоит по просьбе агента", self.reward_config.leader_stop_penalty)
        else:
            res_reward += self.reward_config.leader_movement_reward
            print("Лидер идёт по маршруту", self.reward_config.leader_movement_reward)
        
        if self.is_in_box and self.is_on_trace:
            res_reward += self.reward_config.reward_in_box
            print("В коробке на маршруте.", self.reward_config.reward_in_box)
        elif self.is_in_box:
            # в пределах погрешности
            res_reward += self.reward_config.reward_in_dev
            print("В коробке, не на маршруте", self.reward_config.reward_in_dev)
        elif self.is_on_trace:
            res_reward += self.reward_config.reward_on_track
            print("на маршруте, не в коробке", self.reward_config.reward_on_track)
        else:
            if self.step_count > self.warm_start:
                res_reward += self.reward_config.not_on_track_penalty
            print("не на маршруте, не в коробке", self.reward_config.not_on_track_penalty)
        
        if sum(abs(self.agent_pos - self.leader.cur_pos)) <=self.min_distance:
            res_reward += self.reward_config.too_close_penalty 
            print("Слишком близко!", self.reward_config.too_close_penalty)
        
        if self.crash:
            res_reward += self.reward_config.crash_penalty
            print("АВАРИЯ!", self.reward_config.crash_penalty)
        
        return res_reward
        
    
    
    
class FollowTheLeaderEnv20x20_forward(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__()

class FollowTheLeaderEnv20x20_corner(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy="corner_turn")
        
class FollowTheLeaderEnv20x20_curve(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy="serpentine")#, max_step_size=40)
        
class FollowTheLeaderEnv50x50_curve(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy="curve_turn", size=50)
        
        
register(
    id='MiniGrid-FollowTheLeader-forward-20x20-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv20x20_forward'
)

register(
    id='MiniGrid-FollowTheLeader-corner-20x20-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv20x20_corner'
)

register(
    id='MiniGrid-FollowTheLeader-curve-20x20-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv20x20_curve'
)



#TODO:
# скорость агента;
# адекватное задание маршрутов ведущего;
# корректное отображение границ;
# иные стратегии движения;
# препятствия;
# диалог.