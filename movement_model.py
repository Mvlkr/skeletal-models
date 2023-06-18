from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import math

# УБРАТЬ ВСЕ ПАРАМЕТРЫ!!!!!!!
# ВВЕСТИ НАПРАВЛЕНИЕ ДВИЖЕНИЯ!!!!!

# 4 возможные состояния модели
class States(Enum):
    On_ground = 1
    Right_up = 2
    Left_up = 3
    Both_up = 4


# 7 возможных переходов модели
class Transitions(Enum):
    Same_condition = 0
    Left_going_up = 1
    Right_going_up = 2
    Both_going_up = 3
    Left_going_down = 4
    Right_going_down = 5
    Both_going_down = 6

class Step(Enum):
    Left = 0
    Right = 1

'''
    Опишем разрешенные переходы для каждого из состояний
    имеется предположение, что в спорте положение ног игроков меняется очень часто => для смены положения должны быть
    достаточно маленькие вероятность, а для нахождения в том же положение, достаточно большая уверенность.
    Мне кажется, что выходы из состояния должны быть равновероятными
'''

allowed_transitions = {
    States.On_ground: {'a':[Transitions.Left_going_up,
                       Transitions.Right_going_up, Transitions.Both_going_up], 'p':[0.4, 0.4, 0.2]},
    States.Right_up: {'a':[Transitions.Left_going_up, Transitions.Right_going_down], 'p':[0.1, 0.9]},
    States.Left_up: {'a':[Transitions.Right_going_up, Transitions.Left_going_down], 'p':[0.1, 0.9]},
    States.Both_up: {'a':[Transitions.Left_going_down,
                     Transitions.Right_going_down, Transitions.Both_going_down], 'p':[0.2, 0.2, 0.6]}
}

# allowed_states[transition][prev_state] = new_state
allowed_states = {
    Transitions.Same_condition: {States.On_ground: States.On_ground,
                                 States.Right_up: States.Right_up, States.Left_up: States.Left_up,
                                 States.Both_up: States.Both_up},
    Transitions.Left_going_up: {States.On_ground: States.Left_up, States.Right_up: States.Both_up},
    Transitions.Right_going_up: {States.On_ground: States.Right_up, States.Left_up: States.Both_up},
    Transitions.Both_going_up: {States.On_ground: States.Both_up},
    Transitions.Left_going_down: {States.Both_up: States.Right_up, States.Left_up: States.On_ground},
    Transitions.Right_going_down: {States.Both_up: States.Left_up, States.Right_up: States.On_ground},
    Transitions.Both_going_down: {States.Both_up: States.On_ground}
}

# coordinate_change[prev_state][new_state] = change of coord (left, right, both, same)
coordinate_change = {
    States.Both_up: {States.On_ground: ['left', 'right', 'both'],
                     States.Right_up: ['left'], States.Left_up: ['right']},
    
    States.Right_up: {States.On_ground: ['right']},
    States.Left_up: {States.On_ground: ['left']}
}


@dataclass
class PlayGround:
    type: str = "Поле для Бадминтона"
    width: int = 520  # y
    length: int = 1340  # x
    net: bool = True
    net_pos: int = 670
        
        
# Опишем границы нахождения в каждом из состояний (критерии остановки нахождения в состоянии)
@dataclass
class Restrictions:
    min_y: int = 0
    max_y: int =  PlayGround.width 
    jump_time: int = 5
    left_up_time: int = 2
    right_up_time: int = 2
    distance: int = 120
    min_x: int = 0
    max_x: int = PlayGround.net_pos
   # shift_x: int = 35
   # shift_y: int = 50
        

class Player():
    def __init__(self, env, team, num):
        self.num = num
        self.env = env
        self.team = team
        self.state = States.On_ground  # начальное положение всегда на земле
        self.tr = None
        self.step = None # какой ногой был сделан шаг
        self.prev_state = None
        # будем хранить положение ног для каждого тика
        self.cond_left = [0]  # 1-> нога в воздухе, 0 -> нога на земле
        self.cond_right = [0]
        self.state_counter = 0
        self.leg_distance = 0  # сначала инизиализируем нулевое расстояние между ногами
        # данное положение ног
        self.curr_left_y= np.random.randint(1, PlayGround.width)
        if (self.team == 1): # 1 -> before net; 2 -> after net
            self.curr_left_x = np.random.randint(1, PlayGround.net_pos - 31) 
        else:
            self.curr_left_x = np.random.randint(PlayGround.net_pos + 1, PlayGround.length - 31)
        self.curr_right_x = self.curr_left_x + np.random.randint(1, 30)
        self.curr_right_y = self.curr_left_y + np.random.randint(1, 30)
        # направление (будем выбирать рандомно)
        self.direction = np.random.choice([-1,1])
        self.direction_y = np.random.choice([-1,1])
        # координаты положения ног
        self.left_x = [self.curr_left_x]
        self.left_y = [self.curr_left_y]
        self.right_x = [self.curr_right_x]
        self.right_y = [self.curr_right_y]

    def update_position(self):
        if self.state == States.Both_up:
            self.cond_left.append(1)
            self.cond_right.append(1)
        elif self.state == States.On_ground:
            self.cond_left.append(0)
            self.cond_right.append(0)
        elif self.state == States.Right_up:
            self.cond_left.append(0)
            self.cond_right.append(1)
        elif self.state == States.Left_up:
            self.cond_left.append(1)
            self.cond_right.append(0)
        yield self.env.timeout(1)

    def check_state(self):
        if (self.state == States.Both_up and self.state_counter < Restrictions.jump_time) or \
                (self.state == States.Right_up and self.state_counter < Restrictions.right_up_time) or \
                (self.state == States.Left_up and self.state_counter < Restrictions.left_up_time):
            self.state_counter += 1
            self.tr = Transitions.Same_condition
            self.state = self.state
        else:
            d = allowed_transitions[self.state]
            self.tr = np.random.choice(**d)
            self.state = allowed_states[self.tr][self.state]
            self.state_counter = 0

    def select_state(self):
        probability = np.random.random()
        if probability > 0.9 or probability < 0.1: # скорее всего останемся в том же положении
            self.check_state()
        else:
            if self.state == States.On_ground and self.tr != None:
                if self.prev_state == States.Left_up:
                    self.step = Step.Left 
                elif self.prev_state == States.Right_up:
                    self.step = Step.Right
            else:
                self.step = None
            d = allowed_transitions[self.state]
            if self.step != None:
                if self.step == Step.Left:
                    d['p'][1] = 0.9
                    d['p'][0] = 0.05
                    d['p'][2] = 0.05
                else:
                    d['p'][0] = 0.9
                    d['p'][1] = 0.05
                    d['p'][2] = 0.05
            self.tr = np.random.choice(**d)
            self.prev_state = self.state
            self.state = allowed_states[self.tr][self.state]
            self.state_counter = 0
        self.leg_position()
        yield self.env.process(self.update_position())

    '''
      Положение ног может меняться после прыжка, либо после поднятия одной из ног
      state.Both_up -> положение ног меняется (становится одинаковым либо одна нога находится чуть спереди), положение ног не меняется  
      states.Left_up/states.Right_up -> положение ног остается прежним, либо одна из ног уходит вперед
      Положение ног меняется только для переходных состояний
    '''
    # fl == TRUE -> x ; fl = FALSE -> y
    def check_restr(self, fl, shift_l, shift_r):
        if fl:
            left = self.curr_left_x
            right = self.curr_right_x
            min_restr = 0
            max_restr = PlayGround.net_pos
            if self.team == 2:
                left -= Restrictions.max_x
                right -= Restrictions.max_x
        else:
            min_restr = 0
            max_restr = PlayGround.width
            left = self.curr_left_y
            right = self.curr_right_y

        if min_restr > left + shift_l:
            dist = min_restr + np.random.randint(2,15)
            shift_l = dist - (left + shift_l)

        if min_restr > right + shift_r:
            dist = min_restr + np.random.randint(2,15)
            shift_r = dist - (right + shift_r)

        if max_restr < left + shift_l:
            dist = max_restr - np.random.randint(2,15)
            shift_l = dist - (left + shift_l)

        if max_restr < right + shift_r:
            dist = min_restr - np.random.randint(2,15)
            shift_r = dist - (right + shift_r)

        return shift_l, shift_r

    def check_borders_x(self, shift, state):
        #assert self.curr_right_x > 0 or self.curr_left_x > 0, "Выход за пределы поля по x"
        flag = False
        probability = np.random.random()
        if probability >= 0.95:
            self.direction *= -1
        left = self.curr_left_x
        right = self.curr_right_x
        if self.team == 2:
            left -= Restrictions.max_x
            right -= Restrictions.max_x
        if state == States.Both_up:
            noise_l = np.random.randint(-5,5)
            noise_r = np.random.randint(-5,5)
        else:
            noise_l, noise_r = 0, 0
        shift_l = self.direction * (shift + noise_l)
        if (Restrictions.min_x < left + shift_l) and (Restrictions.max_x > left + shift_l):
            if abs(Restrictions.min_x - (left + shift_l)) < 5 or abs(Restrictions.max_x - (left + shift_l)) < 5:
                self.direction *= -1
                flag = True # alredy changed
        else:
            while (Restrictions.min_x < left + shift_l) and (Restrictions.max_x > left + shift_l):
                if abs(shift_l) == 0:
                    self.direction *= -1
                    break
                else: 
                    shift_l = self.direction*(abs(shift_l) - 1)
        shift_r = self.direction * (shift + noise_r)
        if (Restrictions.min_x < right+ shift_r) and (Restrictions.max_x > right + shift_r):
            if (abs(Restrictions.min_x - (right + shift_r)) < 5 or abs(Restrictions.max_x - (right + shift_r)) < 5) and flag != True:
                self.direction *= -1
        else:
            while (Restrictions.min_x < right + shift_r) and (Restrictions.max_x > right + shift_r):
                if abs(shift_r) == 0:
                    self.direction *= -1
                    break
                else: 
                    shift_r = self.direction*(abs(shift_r) - 1)
        if state == States.Both_up:
            return shift_l, shift_r
        elif state == States.Right_up:
            return 0, shift_r
        elif state == States.Left_up:
            return shift_l, 0
    
    def check_borders_y(self, shift, state):
        #assert self.curr_right_y > 0 or self.curr_left_y > 0, "Выход за пределы поля по y"
        flag = False
        probability = np.random.random()
        if probability >= 0.8: # направление будем менять чаще
            self.direction_y *= -1
        left = self.curr_left_y
        right = self.curr_right_y
        if state == States.Both_up:
            noise_l = np.random.randint(-5,5)
            noise_r = np.random.randint(-5,5)
        else:
            noise_l, noise_r = 0, 0
        shift_l = self.direction_y * (shift + noise_l)
        if (Restrictions.min_y < left + shift_l) and (Restrictions.max_y > left + shift_l):
            if abs(Restrictions.min_y - (left + shift_l)) < 5 or abs(Restrictions.max_y - (left + shift_l)) < 5:
                self.direction_y *= -1
                flag = True # alredy changed
        else:
            while (Restrictions.min_y < left + shift_l) and (Restrictions.max_y > left + shift_l):
                if abs(shift_l) == 0:
                    self.direction_y *= -1
                    break
                else: 
                    shift_l = self.direction_y * (abs(shift_l) - 1)
        shift_r = self.direction_y * (shift + noise_r)
        if (Restrictions.min_y < right+ shift_r) and (Restrictions.max_y > right + shift_r):
            if (abs(Restrictions.min_y - (right + shift_r)) < 5 or abs(Restrictions.max_y - (right + shift_r)) < 5) and flag != True:
                self.direction_y *= -1
        else:
            while (Restrictions.min_y < right + shift_r) and (Restrictions.max_y > right + shift_r):
                if abs(shift_r) == 0:
                    self.direction_y *= -1
                    break
                else: 
                    shift_r = self.direction_y*(abs(shift_r) - 1)
        if state == States.Both_up:
            return shift_l, shift_r
        elif state == States.Right_up:
            return 0, shift_r
        elif state == States.Left_up:
            return shift_l, 0

    # переписать с уменьшением сдвига!!
    def check_distance(self):
        dist = math.dist([self.curr_left_x, self.curr_left_y], [self.curr_right_x, self.curr_right_y])
        #print("NORM DIST", dist)

    # ищем сдвиг положительный сдвиг направление будем определять в другой функции
    def leg_position(self):
        if self.state == States.On_ground and self.prev_state != None:
            shift_x = np.random.randint(0, 60)
            shift_y = np.random.randint(0, 50)
            #print("FOR ERROR",self.prev_state, "shift_x = ",shift_x, "shift_y", shift_y)
            shift_l_x, shift_r_x = self.check_borders_x(shift_x, self.prev_state)
            self.curr_left_x += shift_l_x
            self.curr_right_x += shift_r_x

            shift_l_y, shift_r_y = self.check_borders_y(shift_y, self.prev_state)
            self.curr_left_y += shift_l_y
            self.curr_right_y += shift_r_y

            self.check_distance()
            self.left_x.append(self.curr_left_x)
            self.left_y.append(self.curr_left_y)
            self.right_x.append(self.curr_right_x)
            self.right_y.append(self.curr_right_y)
