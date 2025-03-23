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
        
        #self.physics_client = p.connect(p.GUI)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)  # Define a gravidade corretamente
        self._reset_simulation()
        
    def render(self):
        """Atualiza a simulação na tela"""
        p.stepSimulation()  # Avança a simulação

    
    def _reset_simulation(self):
        p.resetSimulation()
        self.robots = []

        # Posicione os robôs e a bola em posições iniciais válidas
        self.robots.append(p.loadURDF("r2d2.urdf", [75, 65, 0]))  # Exemplo de posição inicial
        self.robots.append(p.loadURDF("r2d2.urdf", [100, 50, 0]))  # Exemplo de posição inicial

        self.ball = p.loadURDF("sphere_small.urdf", [75, 65, 0])  # Posição inicial da bola

        # Configurar atrito e massa do robô e da bola
        for robot in self.robots:
            p.changeDynamics(robot, -1, mass=1.0, lateralFriction=0.5, spinningFriction=0.5)  # Atrito no robô
        p.changeDynamics(self.ball, -1, mass=0.047, lateralFriction=0.5, spinningFriction=0.5)  # Atrito na bola

        # Limitar velocidade máxima do robô
        self.max_speed = 100 # Velocidade máxima permitida
    
    def reset(self, seed=None, options=None):
        """Reseta o ambiente e retorna o estado inicial"""
        self._reset_simulation()
        
        state = self._get_observation()
        return state, {}
    
    def _get_observation(self):
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
        for i, robot in enumerate(self.robots):
            vx, vy = action[2*i] * 10, action[2*i+1] * 10
            # Aplica uma força externa ao robô
            p.applyExternalForce(robot, -1, [vx, vy, 0], [0, 0, 0], p.WORLD_FRAME)

        # Verificar e limitar a velocidade do robô
        robot_vel, _ = p.getBaseVelocity(self.robots[0])
        if np.linalg.norm(robot_vel[:2]) > self.max_speed:
            p.resetBaseVelocity(self.robots[0], linearVelocity=[0, 0, 0])

        # Verifica colisões entre robôs e bola
        for robot in self.robots:
            robot_pos, _ = p.getBasePositionAndOrientation(robot)
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
            distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(ball_pos[:2]))

            # Se o robô estiver próximo à bola, aplica uma força à bola
            if distance < 5:  # Distância de colisão
                p.applyExternalForce(self.ball, -1, [vx, vy, 0], [0, 0, 0], p.WORLD_FRAME)

        p.stepSimulation()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = False
        return obs, reward, done, False, {}
        
    
    def _compute_reward(self):
        # Posição e velocidade da bola
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_vel, _ = p.getBaseVelocity(self.ball)
        target_pos = np.array([150, 65])  # Posição do gol adversário

        # Distância da bola até o gol
        distance_ball_to_target = np.linalg.norm(ball_pos[:2] - target_pos)

        # Velocidade da bola em direção ao gol
        ball_vel_toward_target = np.dot(ball_vel[:2], (target_pos - ball_pos[:2])) / distance_ball_to_target

        # Recompensa principal: incentivar a bola a se mover em direção ao gol
        reward = -distance_ball_to_target + 10 * ball_vel_toward_target  # Aumente o peso da velocidade

        # Penalizar mudanças bruscas na velocidade do robô (suavidade)
        if hasattr(self, 'last_robot_vel'):
            robot_vel, _ = p.getBaseVelocity(self.robots[0])
            vel_change = np.linalg.norm(np.array(robot_vel[:2]) - np.array(self.last_robot_vel[:2]))
            reward -= 0.1 * vel_change  # Penalidade por mudanças bruscas
        self.last_robot_vel, _ = p.getBaseVelocity(self.robots[0])

        return reward
    
    def render(self, mode='human'):
        """Pode ser melhorado com visualização 2D futuramente"""
        pass  # Aqui pode-se implementar uma visualização em pygame ou matplotlib
    
    def close(self):
        p.disconnect()