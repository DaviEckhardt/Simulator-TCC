import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from ambiente import VSSSEnv

env = VSSSEnv()

space = [
    Real(0, 15, name='Kp'),
    Real(0, 1, name='Kd'),
    Real(0, 0.1, name='Ki')
]

initial_parameters = [6.0, 0.9, 0.01]

@use_named_args(space)
def evaluate_parameters(Kp, Kd, Ki):
    global env
    
    observation, _ = env.reset()

    pid_params = {'Kp': Kp, 'Kd': Kd, 'Ki': Ki}

    total_reward = 0
    for _ in range(1000):
        action = pid_controller(observation, pid_params)
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return -total_reward

def pid_controller(observation, pid_params):
    Kp, Kd, Ki = pid_params['Kp'], pid_params['Kd'], pid_params['Ki']
    robot_pos = observation[:2]
    ball_pos = observation[2:]

    error = np.array(ball_pos) - np.array(robot_pos)

    action = Kp * error

    action = np.clip(action, -1, 1)
    return action

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