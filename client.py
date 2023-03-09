import connection as cn
import random
import os
import numpy as np

def get_values(): # Função que lê os valores iniciais da Q-table
    arr = []
    with open(f'{os.getcwd()}/resultado.txt') as file: 
        lines = file.readlines()
        for (idx, line) in enumerate(lines):
            values = line.replace('\n', '').split(' ')
            values = [float(n) for n in values]
            arr.append(values)
    return arr

def write_table(_matriz): # Função que atualiza os valores da Q-table no arquivo .txt
    with open(f'{os.getcwd()}/resultado.txt', 'w') as file:
        text = ''

        for plataform in _matriz:
            text += f'{round(plataform[0], 6)} {round(plataform[1], 6)} {round(plataform[2], 6)}\n'

        file.write(text)

def choose_action() -> str: # Função que escolhe uma ação aleatória
    return random.choice([0, 1, 2])

def get_plataform(_state): # Função que converte a plataforma e a direção de binário para inteiro
    plataform = int(_state[2:-2], 2) 
    direction = int(_state[-2:], 2)

    _direction_plataform = plataform * 4 + direction
    
    return _direction_plataform

def best_action(_directions): # Função que retorna o índice do elemento de maior valor de ação
    return _directions.index(max(_directions[0], _directions[1], _directions[2]))

if __name__ == '__main__':
    sckt = cn.connect(2037)
    matrix = get_values()
    
    # Hiperparâmetros
    alpha = 0.6   # Taxa de aprendizagem
    gamma = 0.9   # Fator de desconto
    
    # Ações possíveis
    dic_actions = {0: 'left',
                  1: 'right',
                  2: 'jump'}
    
    # Estados iniciais
    state = '0b0000000'
    last_state = '0b0000000'
    
    # Variáveis usadas no decaimento exponencial
    exploration_prob = 1
    exploration_decreasing_decay = 0.001
    min_exploration_prob = 0.01

    # Loop principal da execução
    while True:
        for e in range(10000): 
            for i in range(100):
                
                # Escolha randômica da ação através de decaimento exponencial
                if random.uniform(0,1) < exploration_prob:
                    action = choose_action()
                else:
                    direction_plataform = get_plataform(last_state)
                    directions = matrix[direction_plataform]
                    action = best_action(directions)
                    
                state, reward = cn.get_state_reward(sckt, dic_actions[action]) # Ação que retorna o estado e a recompensa resultantes

                # Conversão de binário para inteiro
                plataform = get_plataform(state)
                last_plataform = get_plataform(last_state)

                # Equação de atualização 
                matrix[last_plataform][action] = (1 - alpha) * matrix[last_plataform][action] + alpha * (reward + gamma * (max(matrix[plataform])))

                last_state = state

                write_table(matrix) # Atualização da Q-table
            
            # Escolha entre uma constante (0.01) e o decaimento exponencial de exploration_prob, a fim de setar um limite inferior
            exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay * e))           