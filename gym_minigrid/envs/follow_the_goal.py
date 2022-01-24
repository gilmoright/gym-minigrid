from gym_minigrid.minigrid import *
from gym_minigrid.register import register
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
            leader_start_dir=0,
            n_obstacles=1,
            min_distance = 1,
            max_distance = 4,
            max_dev = 1,
            warm_start = 5,
            movement_strategy = "forward", 
            max_step_size = None,
            #available_actions = ["forward", "turn_left", "turn_right", "left", "right", "message_stop", "speed_up", "speed_down"] # пока не используется
            
    ):
        
        self.agent_start_pos = np.array(agent_start_pos,dtype=int)
        self.agent_start_dir = agent_start_dir
        
        self.leader_start_pos = np.array(leader_start_pos,dtype=int)
        self.leader_trace = list()
        
        self.leader_movement_strategy = self.movement_strategy_generate(movement_strategy)
        self.stop_signal = False
        
        if max_step_size is None:
            max_step_size = len(self.leader_movement_strategy)
        
        
        # Reduce obstacles if there are too many
        self.n_obstacles = n_obstacles
        
        self.leader_trace = []
        
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_dev = max_dev
        self.movement_strategy = movement_strategy
        
        self.warm_start = warm_start
        self.accumulated_reward = 0
        
        
        super().__init__(
            grid_size=size,
            max_steps=max_step_size,#4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        
        self.leader_step = 0
        
        # Allow only 3 actions permitted: left, right, forward
        # + сигнал подождать
        self.action_space = spaces.Discrete(self.actions.toggle + 1)
        self.reward_range = (-100, 100)
        
        self.reset()
        
    
    
    def reset(self):
        super().reset()
        
        self.leader_step = 0
        
        self.leader_trace = list()
        
        self.leader_movement_strategy = self.movement_strategy_generate(self.movement_strategy)
        self.stop_signal = False
        
        
        self.accumulated_reward = 0
        
        
        
        
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
        
    def is_agent_in_box(self):
        
        closest_trace_distance = 1000
        is_on_track = False
        
        for cur_point_nb, cur_trace_point in enumerate(self.leader_trace):
            
            cur_trace_distance = self._coords_diff(cur_trace_point,self.agent_pos)
            
            if cur_trace_distance < closest_trace_distance:
                closest_trace_distance = cur_trace_distance
                closest_trace_point_nb = cur_point_nb
            
            if cur_trace_distance < self.max_dev:
                is_on_track=True
        
        
        
        is_distance_safe = False
        
        #distance = len(self.leader_trace[closest_trace_point_nb:])+closest_trace_distance
        
        distance = self._coords_diff(self.agent_pos, self.leader.cur_pos)
        if self.min_distance < distance < self.max_distance:
            is_distance_safe = True
        
        return is_on_track and is_distance_safe
    
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        #self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = np.array(self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        #Сюда ведущего добавлять
        self.leader = Ball()
        self.put_obj(self.leader, *self.leader_start_pos)

        self.mission = "Следуй за лидером, минимальная дистанция = {0}, максимальная = {1},\n отклонение от маршрута = {2},шагов без штрафа = {3}.".format(self.min_distance, self.max_distance, self.max_dev, self.warm_start)
        
#         self.stop_signal = False
    
    
    def step(self, action):
        reward = 0
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'floor'

        
        #Здесь определяем движение Ведущего
        old_pos = self.leader.cur_pos
        self.leader_trace.append(old_pos)
        
        # обработка стоп-сигнала и движение ведущего
        if self.stop_signal:
            leader_movement = np.zeros(2,dtype=int)
            reward -= 1
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
            reward += 1
        
        self.put_obj(self.leader,*(old_pos+leader_movement))
        
        
        
        size_of_reward_box = self.max_distance - self.min_distance
        unique_points_iter = 0
        
        #self.cur_bounding_box = list()
        
        is_on_trace = False # находится ли агент на маршруте;
        is_in_box = False # находится ли агент в "коробке";
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
                        is_on_trace = True


                if np.any(cur_leader_trace_point!=prev_leader_trace_point):
                    unique_points_iter += 1

                if (unique_points_iter > self.min_distance) and (unique_points_iter <= self.max_distance):
                    self.put_obj(Floor("red"), *cur_leader_trace_point)
    #                 self.cur_bounding_box.append(cur_leader_trace_point)

                    trace_diff = sum(abs(self.agent_pos - cur_leader_trace_point))
                    if (trace_diff==0) or (trace_diff <= self.max_dev and not is_on_trace):
                        is_in_box = True

        
        # Обработка действий агента будет в этой функции
        done = self._agent_action_processing(action)
        
        
        if is_in_box and is_on_trace:
            reward += 1
            print("В коробке на маршруте.")
        elif is_in_box:
            # в пределах погрешности
            reward += 0.5
            print("В коробке, не на маршруте")
        elif is_on_trace:
            reward += 0.1
            print("на маршруте, не в коробке")
        else:
            if self.step_count > self.warm_start:
                reward += -1
            print("не на маршруте, не в коробке")
        
        if sum(abs(self.agent_pos - self.leader.cur_pos)) <=self.min_distance:
            reward -= 5 
            print("Слишком близко!")
        

        
        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -10
            print("Авария!")
            done = True
        #    return obs, reward, done, info
        
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

    
    
    def movement_strategy_generate(self, strategy_name):#, stop_on_block = True):
        with open("movement_strategies/{}.txt".format(strategy_name), "r") as strat_file:
            strategy_commands = strat_file.readlines()
        
        list_commands_by_step = list()
        for cur_command in strategy_commands:
            t_str = cur_command.strip("\n")
            step_nb, first_coord, sec_coord = cur_command.split(";")
            
            for i in range(1,int(step_nb)):
                list_commands_by_step.append(np.array([int(first_coord),int(sec_coord)], dtype=int))
        
        return list_commands_by_step
        
#         leader_movement = np.zeros(2,dtype=int)
        
#         if step < 10:
#             leader_movement[1]+=1
#         else:
#             leader_movement[0]+=1
        
#         # отслеживание возможности продолжения движения
#         # добавить огибание
#         if self.grid.get(*self.leader.cur_pos+leader_movement) not in {None, "floor"}:
#             if stop_on_block:
#                 return np.zeros(2,dtype=int)
#             else:
#                 raise NotImplementedError()
#                 # пробуем втупую обойти стену
#                 # уверен, можно проще
#                 if leader_movement[0] == 1:
#                     if self.grid.get(*self.leader.cur_pos+[0,1]) not in {None, "floor"}:
                        
                
            
#         else:
#             return leader_movement
        
    
    def _agent_action_processing(self, action):
        done = False
        
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
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

            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Stop_signal
        elif action == self.actions.toggle:
            self.stop_signal = not self.stop_signal
            print("стоп-сигнал -- {}".format(self.stop_signal))
            
        return done
    
    
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