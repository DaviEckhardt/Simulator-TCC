import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

class VSS2DEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Definição do espaço de ação (velocidades dos motores dos robôs)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))  # 2 robôs, cada um com 2 rodas
        
        # Definição do espaço de observação (posição e velocidade dos robôs e bola)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -10, -10] * 3),  # Para 2 robôs + bola (posição X, Y e velocidades Vx, Vy)
            high=np.array([150, 130, 10, 10] * 3),
            dtype=np.float32
        )
        
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        self._reset_simulation()
    
    def _reset_simulation(self):
        """Inicializa o ambiente com robôs e bola"""
        p.resetSimulation()
        self.robots = []
        
        # Criando os robôs (caixas simples para representar os jogadores)
        for _ in range(2):
            self.robots.append(p.loadURDF("r2d2.urdf", [np.random.uniform(0, 150), np.random.uniform(0, 130), 0]))
        
        # Criando a bola
        self.ball = p.loadURDF("sphere_small.urdf", [75, 65, 0])
    
    def reset(self, seed=None, options=None):
        """Reseta o ambiente e retorna o estado inicial"""
        self._reset_simulation()
        
        state = self._get_observation()
        return state, {}
    
    def _get_observation(self):
        """Coleta a posição e velocidade dos robôs e da bola"""
        obs = []
        for robot in self.robots:
            pos, _ = p.getBasePositionAndOrientation(robot)
            vel, _ = p.getBaseVelocity(robot)
            obs.extend([pos[0], pos[1], vel[0], vel[1]])
        
        pos, _ = p.getBasePositionAndOrientation(self.ball)
        vel, _ = p.getBaseVelocity(self.ball)
        obs.extend([pos[0], pos[1], vel[0], vel[1]])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Executa uma ação no ambiente"""
        for i, robot in enumerate(self.robots):
            vx, vy = action[2*i:2*i+2]
            p.resetBaseVelocity(robot, linearVelocity=[vx, vy, 0])
        
        p.stepSimulation()
        
        obs = self._get_observation()
        reward = self._compute_reward()
        done = False  # Definir critério de término depois
        return obs, reward, done, False, {}
    
    def _compute_reward(self):
        """Define um sistema de recompensas básico"""
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        reward = -np.linalg.norm(np.array(ball_pos[:2]) - np.array([150, 65]))  # Aproximar da meta adversária
        return reward
    
    def render(self, mode='human'):
        """Pode ser melhorado com visualização 2D futuramente"""
        pass  # Aqui pode-se implementar uma visualização em pygame ou matplotlib
    
    def close(self):
        p.disconnect()