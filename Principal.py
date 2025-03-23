import numpy as np
from scipy.optimize import minimize
import gymnasium as gym
from Ambiente import VSS2DEnv  # Substitua pelo nome real do seu ambiente

# Função para rodar o ambiente e calcular o erro total com um PID específico
def simulacao_pid(params):
    kp, ki, kd = params
    
    env = VSS2DEnv()  # Inicializa o ambiente
    obs, _ = env.reset()
    
    erro_total = 0
    erro_anterior = 0
    integral = 0
    
    for _ in range(100):  # Número de passos por episódio
        posicao_bola = obs[0:2]  # Suponha que os primeiros 2 valores da observação sejam (x, y) da bola
        setpoint = np.array([75, 65])  # Centro do gol adversário (exemplo)
        erro = setpoint - posicao_bola
        
        integral += erro
        derivativo = erro - erro_anterior
        controle = kp * erro + ki * integral + kd * derivativo
        
        # Ação do robô: usa o controle PID nas rodas (ajuste conforme necessário)
        action = np.clip(controle, -1, 1)  # Ajuste para seu espaço de ações
        obs, _, _, _, _ = env.step(action)
        
        erro_anterior = erro
        erro_total += np.linalg.norm(erro)  # Soma dos erros ao longo do episódio
    
    env.close()
    return erro_total  # Queremos minimizar o erro total

# Otimização dos parâmetros PID
resultado = minimize(simulacao_pid, x0=[1, 0, 0], bounds=[(0, 10), (0, 1), (0, 1)])

print("Melhores parâmetros PID encontrados:")
print(f"Kp: {resultado.x[0]:.4f}, Ki: {resultado.x[1]:.4f}, Kd: {resultado.x[2]:.4f}")
