import connection as cn
import random
import os
import numpy as np

def get_values():
    arr = []
    with open(f'{os.getcwd()}/resultado.txt') as file: 
        lines = file.readlines()
        for (idx, line) in enumerate(lines):
            values = line.replace('\n', '').split(' ')
            values = [float(n) for n in values]
            arr.append(values)
    return arr

def write_table(_matriz):
    with open(f'{os.getcwd()}/resultado.txt', 'w') as file:
        text = ''

        for plataform in _matriz:
            text += f'{round(plataform[0], 6)} {round(plataform[1], 6)} {round(plataform[2], 6)}\n'

        file.write(text)

def choose_action() -> str:
    return random.choice([0, 1, 2])

def get_plataform(_state):
    plataform = int(_state[2:-2], 2) # convert to base 2
    direction = int(_state[-2:],2)

    _direction_plataform = plataform * 4 + direction
    
    return _direction_plataform

def best_action(_directions):
    return _directions.index(max(_directions[0], _directions[1], _directions[2]))

if __name__ == '__main__':
    sckt = cn.connect(2037)
    matrix = get_values()
    
    # Hiperparâmetros
    alpha = 0.5   # taxa de aprendizagem
    gamma = 0.9   # fator de desconto
    
    ##################################
    dic_actions = {0: 'left',
                  1: 'right',
                  2: 'jump'}
    state = '0b0000000'
    last_state = '0b0000000'
    ##################################
    
    exploration_prob = 1
    exploration_decreasing_decay = 0.001
    min_exploration_prob = 0.01

    while True:
        for e in range(10000): 
            for i in range(100):
                
                # Escolha da ação feita pelo agente de forma randômica
                if random.uniform(0,1) < exploration_prob:
                    action = choose_action()
                else:
                    direction_plataform = get_plataform(last_state)
                    directions = matrix[direction_plataform]
                    action = best_action(directions)
                    
                state, reward = cn.get_state_reward(sckt, dic_actions[action])

                plataform = get_plataform(state)
                last_plataform = get_plataform(last_state)

                matrix[last_plataform][action] = (1 - alpha) * matrix[last_plataform][action] + alpha * (reward + gamma * (max(matrix[plataform])))

                last_state = state

                write_table(matrix)
            
            exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * e))
