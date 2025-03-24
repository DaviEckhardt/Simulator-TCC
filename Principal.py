import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from ambiente import VSSSEnv  

env = VSSSEnv()

space = [
    Real(0, 15, name='Kp'),    # Aumentado para resposta mais rápida
    Real(0, 1, name='Kd'),     # Aumentado para melhor amortecimento
    Real(0, 0.1, name='Ki')    # Aumentado para correção mais efetiva
]

initial_parameters = [8.0, 0.9, 0.05]  # Valores mais agressivos

@use_named_args(space)
def evaluate_parameters(Kp, Kd, Ki):
    global env  
    observation, _, _, _, _ = env.reset()
    
    # Atualizar parâmetros do PID no ambiente
    env.Kp = Kp
    env.Kd = Kd
    env.Ki = Ki

    total_reward = 0
    for _ in range(1000):  
        # Passar a observação como ação (o ambiente agora calcula o PID internamente)
        observation, reward, done, _, _ = env.step(observation)
        total_reward += reward

        if done:
            break

    return -total_reward

result = gp_minimize(
    func=evaluate_parameters,  
    dimensions=space,          
    n_calls=50,                
    n_random_starts=10,        
    x0=initial_parameters      
)

env.close()

print("Melhores parâmetros encontrados:")
print(f"Kp: {result.x[0]}, Kd: {result.x[1]}, Ki: {result.x[2]}")
print(f"Melhor recompensa: {-result.fun}")