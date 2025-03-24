import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from Ambiente import VSSSEnv  

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

    pid_params = {'Kp': Kp, 'Kd': Kd, 'Ki': Ki}

    integral = np.array([0.0, 0.0])
    last_error = np.array([0.0, 0.0])
    last_action = np.array([0.0, 0.0])

    total_reward = 0
    for _ in range(1000):  
        action = pid_controller(observation, pid_params, integral, last_error, last_action)
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return -total_reward

def pid_controller(observation, pid_params, integral, last_error, last_action):
    Kp, Kd, Ki = pid_params['Kp'], pid_params['Kd'], pid_params['Ki']
    robot_pos = observation[:2]
    ball_pos = observation[2:]
    
    # Calcular direção para a bola
    direction_to_ball = ball_pos - robot_pos
    distance_to_ball = np.linalg.norm(direction_to_ball)
    direction_to_ball = direction_to_ball / (distance_to_ball + 1e-6)
    
    # Calcular direção do gol
    goal_direction = env.goal_pos - ball_pos
    goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
    
    # Calcular posição alvo atrás da bola
    target_pos = ball_pos - goal_direction * (0.025 + 0.04 + 0.1)  # raio da bola + raio do robô + margem
    
    # Calcular ângulo atual do robô
    robot_direction = np.array([np.cos(env.robot_angle), np.sin(env.robot_angle)])
    
    # Calcular direção para o alvo
    direction_to_target = target_pos - robot_pos
    direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-6)
    
    # Calcular ângulo entre a direção do robô e a direção para o alvo
    dot_product = np.dot(robot_direction, direction_to_target)
    cross_product = np.cross(robot_direction, direction_to_target)
    angle_error = np.arctan2(cross_product, dot_product)
    angle_error = np.mod(angle_error + np.pi, 2 * np.pi) - np.pi
    
    # Calcular ângulo entre a direção do robô e a direção do gol
    dot_product_goal = np.dot(robot_direction, goal_direction)
    cross_product_goal = np.cross(robot_direction, goal_direction)
    angle_to_goal = np.arctan2(cross_product_goal, dot_product_goal)
    angle_to_goal = np.mod(angle_to_goal + np.pi, 2 * np.pi) - np.pi
    
    # Calcular erro de posição
    position_error = target_pos - robot_pos
    
    # Atualizar integral e derivada com suavização
    integral = integral * 0.95 + position_error * 0.01
    derivative = (position_error - last_error) / 0.01
    
    # Fator baseado na distância (ajustado para ser mais agressivo quando longe)
    distance_factor = min(1.0, distance_to_ball / 0.3)  # Reduzido de 0.5 para 0.3
    
    # Calcular velocidade linear usando PID
    linear_velocity = 1.5 * (Kp * np.linalg.norm(position_error) + Ki * np.linalg.norm(integral) + Kd * np.linalg.norm(derivative))
    linear_velocity = min(1.0, linear_velocity)  # Limitar velocidade máxima
    
    # Ajustar velocidade linear baseado no ângulo e distância
    angle_factor = np.cos(angle_error)
    linear_velocity *= max(0.5, angle_factor) * distance_factor  # Aplicar distance_factor
    
    # Calcular velocidade angular usando PID
    if distance_to_ball > 0.2:  # Se estiver longe da bola
        # Priorizar chegar atrás da bola
        angular_velocity = (Kp * angle_error + Kd * (angle_error - last_error[0]) + Ki * integral[0]) * distance_factor
        # Aumentar velocidade linear quando estiver bem alinhado
        if abs(angle_error) < np.pi/6:  # Menos de 30 graus
            linear_velocity *= 1.2
    else:  # Se estiver próximo da bola
        # Priorizar alinhar com o gol
        angular_velocity = 0.5 * (Kp * angle_to_goal + Kd * (angle_to_goal - last_error[1]) + Ki * integral[1])
        linear_velocity *= 0.8
    
    # Usar ré apenas quando estiver muito desalinhado e longe da bola
    if abs(angle_error) > np.pi/2 and distance_to_ball > 0.5:
        linear_velocity *= -1
    
    # Converter velocidades linear e angular em velocidades das rodas
    wheel_separation = env.wheel_separation
    left_speed = linear_velocity - (angular_velocity * wheel_separation / 2)
    right_speed = linear_velocity + (angular_velocity * wheel_separation / 2)
    
    # Normalizar as velocidades
    max_speed = max(abs(left_speed), abs(right_speed))
    if max_speed > 1.0:
        left_speed /= max_speed
        right_speed /= max_speed
    
    # Suavizar as ações
    smoothing_factor = 0.9
    left_speed = smoothing_factor * left_speed + (1 - smoothing_factor) * last_action[0]
    right_speed = smoothing_factor * right_speed + (1 - smoothing_factor) * last_action[1]
    
    # Atualizar última ação e erro
    last_action[:] = [left_speed, right_speed]
    last_error[:] = [angle_error, angle_to_goal]
    
    return np.array([left_speed, right_speed])


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