from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.utils.reward_constructor import Reward
from operator import add
import numpy as np
import os
from itertools import cycle

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
            width=None,
            height=None,
            agent_start_pos=(5,1),
            agent_start_dir=0,
            leader_start_pos=(5,3),
            min_distance = 1,
            max_distance = 4,
            max_dev = 1,
            warm_start = 3,
            movement_strategy = "forward", 
            random_step = 3,
            max_steps = None,
            reward_config = None,
            n_obstacles = 0,
            see_through_walls=True,
            agent_view_size=7,
            seed=42
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
            
        max_steps -- int or None:
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
        # Initialize the RNG
        self.seed(seed=seed)
        self.agent_start_pos = np.array(agent_start_pos,dtype=int)
        self.agent_start_dir = agent_start_dir
        self.leader_start_pos = np.array(leader_start_pos,dtype=int)
        self.verbose = 0
        
        self.n_obstacles = n_obstacles
        
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_dev = max_dev

        # Если задан list, при первом вызове _determine_movement_strategy он будет преобразован в тип cycle
        self.movement_strategy = movement_strategy
        
        if movement_strategy == "random":
            self.random_step = random_step
        
        self.warm_start = warm_start
        self.accumulated_reward = 0
        
        # надо инициализировать перед вызовом random_strategy_generate внутри _determine_movement_strategy
        self.max_steps = max_steps        
        self.leader_movement_strategy = self._determine_movement_strategy()
        if self.max_steps is None:
            self.max_steps = len(self.leader_movement_strategy) + 10  # +10 на остановку лидера
        
        self.simulation_nb = 0

        # Can't set both grid_size and width/height
        if size:
            assert width == None and height == None
            width = size
            height = size
        
        
        # Environment configuration
        self.width = width
        self.height = height
        self.see_through_walls = see_through_walls

        
        
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Window to use for human rendering mode
        self.window = None

        
        self.actions = self.Actions
        self.action_space = spaces.Discrete(self.actions.toggle + 1)
        
        if reward_config:
            self.reward_config = Reward.from_json(reward_config)
        else:
            self.reward_config = Reward()
            
        #self.reward_range = (-100, 100)
        self.reward_range = (-3, 3)
        
        self.reset()
        
        
    
    
    def reset(self):
        if self.verbose >= 1:
            print("===Запуск симуляции номер {}===".format(self.simulation_nb))
            print()
        # Из отцовского класса
        self._gen_grid(self.width, self.height)
        
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        
        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

 
        # Модификация
        self.leader_step = 0
        self.leader_trace = list()
        self.unique_trace_points = list() # используется в отрисовке маршрута и определении бокса для награды;
        
        self.cur_min_border_id = 0 
        self.cur_max_border_id = 0
        
        if self.simulation_nb > 0:
            self.leader_movement_strategy = self._determine_movement_strategy()
        
        self.max_steps = len(self.leader_movement_strategy) + 10 # +10 на 3 остановки и запуска лидера
        self.stop_signal = False
        self.crash = False
        
        self.accumulated_reward = 0
        self.simulation_nb += 1
        
        # Return first observation
        obs = self.gen_obs()
        return obs
        
        
    def _determine_movement_strategy(self):
        if isinstance(self.movement_strategy,list):
            self.movement_strategy = cycle(self.movement_strategy)
            cur_strategy_name = next(self.movement_strategy)
        elif isinstance(self.movement_strategy,cycle):
            cur_strategy_name = next(self.movement_strategy)
        else:
            cur_strategy_name = self.movement_strategy
            
            
        if cur_strategy_name == "random":
            leader_movement_strategy = self.random_walking_strategy_generate()
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
        # Здесь определяем движение Ведущего
        self._leader_movement()
        # Отрисовка маршрута
        self._trace_drawing()
        # Обработка действий агента
        done = self._agent_action_processing(action)
        # Определение местоположения агента относительно маршрута
        self._bounding_box_agent_location()
        #Расчёт награды
        reward = self._reward_computation()
        if self.verbose >= 1:
            print("step", self.step_count, "action", action)
        
        self.accumulated_reward += reward
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        if self.verbose >= 1:
            print("Аккумулированная награда: {}".format(self.accumulated_reward))
            print()
        
        return obs, reward, done, {}# info

    
    
    def movement_strategy_generate(self, strategy_name):
            
        if self.verbose >= 1:
            print("выбранная стратегия движения: ", strategy_name)
        
        with open("{}/../movement_strategies/{}.txt".format(os.path.dirname(os.path.abspath(__file__)), strategy_name)) as strat_file:
            strategy_commands = strat_file.readlines()
        list_commands_by_step = list()
        for cur_command in strategy_commands:
            t_str = cur_command.strip("\n")
            step_nb, first_coord, sec_coord = cur_command.split(";")
            
            for i in range(1,int(step_nb)):
                list_commands_by_step.append(np.array([int(first_coord),int(sec_coord)], dtype=int))
        
        return list_commands_by_step
        
    
    def random_walking_strategy_generate(self):
        list_commands_by_step = list()
        
        for cur_command_nb in range(int(self.max_steps/self.random_step)):
            cur_command = np.array((self._rand_int(-1,2), self._rand_int(-1,2)))
            for i in range(self.random_step):
                list_commands_by_step.append(cur_command.copy())
        
        return list_commands_by_step
    
    
    def _leader_movement(self):
        old_pos = self.leader.cur_pos
        self.leader_trace.append(old_pos)
        
        # обработка стоп-сигнала и движение ведущего
        if self.stop_signal:
            leader_movement = np.zeros(2,dtype=int)
            if self.verbose >= 1:
                print("Ведущий стоит по просьбе агента.")
        else:
            if self.leader_step >= len(self.leader_movement_strategy):
                leader_movement = np.zeros(2,dtype=int)
                if self.verbose >= 1:
                    print("Лидер прибыл в точку назначения.", self.leader_step, self.leader_movement_strategy)
            else:
                leader_movement = self.leader_movement_strategy[self.leader_step]
            
            if self.grid.get(*self.leader.cur_pos+leader_movement) not in {None, "floor"}:
                leader_movement = np.zeros(2,dtype=int)
            
            self.leader_step+=1
        
        self.put_obj(self.leader,*(old_pos+leader_movement))
        
        if len(self.leader_trace) < 2:
            self.unique_trace_points.append(old_pos)
        
        if not np.array_equal(old_pos, self.leader.cur_pos):
            self.unique_trace_points.append(self.leader.cur_pos)
            
            # движение рамок
            if len(self.unique_trace_points[self.cur_min_border_id:-1])-1>self.min_distance:
                self.cur_min_border_id += 1

            if self.cur_min_border_id - self.cur_max_border_id == self.max_distance - self.min_distance:
                self.cur_max_border_id += 1
            if self.verbose >= 1:
                print("borders:",self.cur_max_border_id,self.cur_min_border_id,self.max_distance - self.min_distance)
        
    
#     def movement_strategy_generate(self, strategy_name):#, stop_on_block = True):


    def _trace_drawing(self):
        
        if len(self.leader_trace) == 1:
            self.put_obj(Floor("yellow"), *self.leader_trace[-1])
            
        elif self.leader.cur_pos != self.leader_trace[-1]:
        # Лидер двинулся (прям как я)
            self.put_obj(Floor("yellow"), *self.leader_trace[-1])
            self.put_obj(Floor("yellow"), *self.unique_trace_points[self.cur_max_border_id])
            
            for cur_point in self.unique_trace_points[self.cur_max_border_id:self.cur_min_border_id+1]:
                self.put_obj(Floor("red"), *cur_point) 
            
            if self.cur_min_border_id-self.cur_max_border_id >= self.max_distance - self.min_distance-1:
                
                self.put_obj(Floor("yellow"), *self.unique_trace_points[max([self.cur_max_border_id-1,0])])
                
    
    def _bounding_box_agent_location(self):
        
        self.is_on_trace = False # находится ли агент на маршруте;
        self.is_in_box = False # находится ли агент в "коробке";
        
        for cur_point_id, cur_point in enumerate(self.unique_trace_points):
            if np.array_equal(self.agent_pos,cur_point):
                self.is_on_trace = True
        
        
        if len(self.unique_trace_points) > self.min_distance:
            
             for cur_box_point in self.unique_trace_points[self.cur_max_border_id:self.cur_min_border_id+1]:
                    
                trace_diff = sum(abs(self.agent_pos - cur_box_point))
                
                if (trace_diff==0) or (trace_diff <= self.max_dev and not self.is_on_trace):
                    self.is_in_box = True    
            
                    
        
    
    
    def _agent_action_processing(self, action):
        done = False
        
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        
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
                if self.verbose >= 1:
                    print("Авария!") 
                self.crash = True
                done = True
                
#             Пока лавы нет, это не нужно
#             if fwd_cell != None and fwd_cell.type == 'lava':
#                 done = True

        # Stop_signal
        elif action == self.actions.toggle:
            self.stop_signal = not self.stop_signal
            if self.verbose >= 1:
                print("стоп-сигнал -- {}".format(self.stop_signal))
            
        return done
    
    
    
    def _reward_computation(self):
        # Скорее всего, это можно сделать красивее
        res_reward = 0
        
        if self.stop_signal:
            res_reward += self.reward_config.leader_stop_penalty
            if self.verbose >= 1:
                print("Лидер стоит по просьбе агента", self.reward_config.leader_stop_penalty)
        else:
            res_reward += self.reward_config.leader_movement_reward
            if self.verbose >= 1:
                print("Лидер идёт по маршруту", self.reward_config.leader_movement_reward)
        
        if self.is_in_box and self.is_on_trace:
            res_reward += self.reward_config.reward_in_box
            if self.verbose >= 1:
                print("В коробке на маршруте.", self.reward_config.reward_in_box)
        elif self.is_in_box:
            # в пределах погрешности
            res_reward += self.reward_config.reward_in_dev
            if self.verbose >= 1:
                print("В коробке, не на маршруте", self.reward_config.reward_in_dev)
        elif self.is_on_trace:
            res_reward += self.reward_config.reward_on_track
            if self.verbose >= 1:
                print("на маршруте, не в коробке", self.reward_config.reward_on_track)
        else:
            if self.step_count > self.warm_start:
                res_reward += self.reward_config.not_on_track_penalty
            if self.verbose >= 1:
                print("не на маршруте, не в коробке", self.reward_config.not_on_track_penalty)
        
        
        
        leader_agent_diff_vec = abs(self.agent_pos-self.leader.cur_pos)
    
        # Определяем близость так, а не по расстоянию Миньковского, чтобы ученсть близость по диагонали
        if (leader_agent_diff_vec[0]<=self.min_distance) and (leader_agent_diff_vec[1] <= self.min_distance):
#         if sum(abs(self.agent_pos - self.leader.cur_pos)) <= self.min_distance:
            res_reward += self.reward_config.too_close_penalty 
            if self.verbose >= 1:
                print("Слишком близко!", self.reward_config.too_close_penalty)
        
        if self.crash:
            res_reward += self.reward_config.crash_penalty
            if self.verbose >= 1:
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
        super().__init__(movement_strategy="serpentine")

class FollowTheLeaderEnv20x20_cycle_all_strats(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy=["serpentine", "corner_turn", "curve_turn", "forward", "round"])

class FollowTheLeaderEnv50x50_curve(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy="curve_turn", size=50)

class FollowTheLeaderEnv50x50_random(FollowTheLeaderEnv):
    def __init__(self):
        super().__init__(movement_strategy="random", max_steps=50, random_step=2, size=50, leader_start_pos=(25, 25), agent_start_pos=(25, 23))
        
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

register(
    id='MiniGrid-FollowTheLeader-curve-50x50-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv50x50_curve'
)

register(
    id='MiniGrid-FollowTheLeader-cycle_all_strats-20x20-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv20x20_cycle_all_strats'
)

register(
    id='MiniGrid-FollowTheLeader-random-50x50-v0',
    entry_point='gym_minigrid.envs:FollowTheLeaderEnv50x50_random'
)


#TODO:
# скорость агента;
# адекватное задание маршрутов ведущего;
# корректное отображение границ;
# иные стратегии движения;
# препятствия;
# диалог.