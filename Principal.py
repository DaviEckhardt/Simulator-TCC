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
    
    print(f"\nIteração atual - Kp: {Kp:.2f}, Kd: {Kd:.2f}, Ki: {Ki:.2f}")

    total_reward = 0
    for _ in range(1000):  
        # Passar a observação como ação (o ambiente agora calcula o PID internamente)
        observation, reward, done, _, _ = env.step(observation)
        total_reward += reward

        if done:
            break
    
    print(f"Recompensa total: {total_reward:.2f}")
    return -total_reward

print("Iniciando otimização...")
print(f"Parâmetros iniciais - Kp: {initial_parameters[0]:.2f}, Kd: {initial_parameters[1]:.2f}, Ki: {initial_parameters[2]:.2f}")

result = gp_minimize(
    func=evaluate_parameters,  
    dimensions=space,          
    n_calls=50,                
    n_random_starts=10,        
    x0=initial_parameters      
)

env.close()

print("\nResultados finais:")
print("Melhores parâmetros encontrados:")
print(f"Kp: {result.x[0]:.2f}, Kd: {result.x[1]:.2f}, Ki: {result.x[2]:.2f}")
print(f"Melhor recompensa: {-result.fun:.2f}")