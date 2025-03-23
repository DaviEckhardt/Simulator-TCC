import numpy as np
from scipy.optimize import minimize
from Ambiente import VSS2DEnv  # Certifique-se de que este é o nome correto da classe

# Criar ambiente
env = VSS2DEnv()

def simulacao_pid(params):
    kp, ki, kd = params
    print(f"Kp: {kp}, Ki: {ki}, Kd: {kd}")

    # Inicializar simulação no ambiente
    obs, _ = env.reset()
    print(f"Estado inicial (obs): {obs}")

    erro_anterior = 0
    integral = 0
    total_recompensa = 0

    for _ in range(100):  # Simula 100 iterações
        # Extrair posições do robô e da bola do estado observado
        robot_pos = np.array([obs[0], obs[1]])  # Posição do robô (X, Y)
        ball_pos = np.array([obs[8], obs[9]])   # Posição da bola (X, Y)
        target_pos = np.array([150, 65])        # Posição do gol adversário

        # Distâncias
        distance_robot_to_ball = np.linalg.norm(robot_pos - ball_pos)
        distance_ball_to_target = np.linalg.norm(ball_pos - target_pos)

        # Ângulo entre o robô, a bola e o gol
        vector_robot_to_ball = ball_pos - robot_pos
        vector_ball_to_target = target_pos - ball_pos
        angle = np.arctan2(vector_robot_to_ball[1], vector_robot_to_ball[0]) - np.arctan2(vector_ball_to_target[1], vector_ball_to_target[0])

        # Erro combinado (normalizado)
        erro = (distance_robot_to_ball + distance_ball_to_target + abs(angle)) / 100

        # Controle PID
        integral += erro
        derivativo = erro - erro_anterior
        erro_anterior = erro

        # Calcular controle para as duas rodas de cada robô
        controle_esq = kp * erro + ki * integral + kd * derivativo
        controle_dir = kp * erro + ki * integral - kd * derivativo

        # Escalonar as ações para evitar valores extremos
        acao_escalonada_esq = controle_esq / 100
        acao_escalonada_dir = controle_dir / 100

        # Garantir que os valores estejam dentro do intervalo permitido
        action = np.clip([acao_escalonada_esq, acao_escalonada_dir, acao_escalonada_esq, acao_escalonada_dir], -1, 1)

        # Envia a ação e obtém o próximo estado
        obs, recompensa, done, _, _ = env.step(action)
        total_recompensa += recompensa
        

        if done:
            break

    print(f"Recompensa total: {total_recompensa}")
    return -total_recompensa  # Minimizar erro => maximizar recompensa negativa

# Otimização com limites corrigidos
resultado = minimize(
    simulacao_pid,
    x0=[1, 0.01, 0.1],  # Valores iniciais de Kp, Ki e Kd
    bounds=[(0, 15), (0, 0.1), (0, 1)],  # Limites para Kp, Ki e Kd
    method='L-BFGS-B',  # Método alternativo
    options={'maxiter': 10, 'eps': 1e-2}  # Aumenta o número de iterações e o tamanho do passo inicial
)

print("Melhores valores encontrados:")
print("Kp:", resultado.x[0])
print("Ki:", resultado.x[1])
print("Kd:", resultado.x[2])