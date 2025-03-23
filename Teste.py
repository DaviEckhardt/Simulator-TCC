import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from scipy.optimize import minimize

class VSSSoccerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.width, self.height = 600, 520
        
        # Definição da ação e estado
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Configuração do ambiente gráfico
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Very Small Size Soccer Simulator")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self._setup_simulation()
    
    def _setup_simulation(self):
        """Cria os objetos da simulação."""
        self.ball = self._create_circle(self.width // 2, self.height // 2, 8, mass=1)
        self.robot = self._create_circle(self.width // 4, self.height // 2, 30, mass=5)
    
    def _create_circle(self, x, y, radius, mass=1):
        """Cria um corpo circular no pymunk."""
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = (x, y)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.8
        self.space.add(body, shape)
        return body
    
    def reset(self, seed=None, options=None):
        """Reseta o ambiente para o estado inicial."""
        super().reset(seed=seed)
        self._setup_simulation()
        return self._get_observation(), {}
    
    def step(self, action):
        """Aplica a ação ao ambiente."""
        force_x = action[0] * 1000
        force_y = action[1] * 1000
        self.robot.apply_force_at_local_point((force_x, force_y), (0, 0))
        
        self.space.step(1 / 60)
        obs = self._get_observation()
        reward = -np.linalg.norm(obs[:2])  # Penaliza distância ao centro
        done = False  # Pode definir uma condição de término
        return obs, reward, done, False, {}
    
    def _get_observation(self):
        """Retorna a posição da bola e do robô normalizada."""
        ball_pos = np.array(self.ball.position) / [self.width, self.height] * 2 - 1
        robot_pos = np.array(self.robot.position) / [self.width, self.height] * 2 - 1
        return np.concatenate([ball_pos, robot_pos])
    
    def render(self):
        """Renderiza o ambiente."""
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(60)
    
    def close(self):
        if self.render_mode == "human":
            pygame.quit()

# ---- Otimização PID ---- #
def simulate_pid(params):
    kp, ki, kd = params
    env = VSSSoccerEnv()
    obs, _ = env.reset()
    total_error = 0
    
    for _ in range(100):
        error = -obs[2:]  # Erro na posição do robô
        integral = np.sum(error)
        derivative = error - obs[2:]
        action = kp * error + ki * integral + kd * derivative
        action = np.clip(action, -1, 1)
        obs, reward, done, _, _ = env.step(action)
        total_error += np.linalg.norm(error)
    
    env.close()
    return total_error

resultado = minimize(simulate_pid, x0=[1, 0, 0], bounds=[(0, 10), (0, 1), (0, 1)])
print(f"Melhores parâmetros PID: Kp={resultado.x[0]:.4f}, Ki={resultado.x[1]:.4f}, Kd={resultado.x[2]:.4f}")

# Teste de simulação com PID otimizado
env = VSSSoccerEnv(render_mode="human")
env.reset()
for _ in range(300):
    obs, _ = env.reset()
    error = -obs[2:]
    integral = np.sum(error)
    derivative = error - obs[2:]
    action = resultado.x[0] * error + resultado.x[1] * integral + resultado.x[2] * derivative
    action = np.clip(action, -1, 1)
    env.step(action)
    env.render()
env.close()
