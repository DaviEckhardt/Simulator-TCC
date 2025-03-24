import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VSSSEnv(gym.Env):
    def __init__(self):
        super(VSSSEnv, self).__init__()
        self.clock = pygame.time.Clock()
        self.ball_velocity = np.array([0.0, 0.0])
        self.friction = 0.99

        self.field_length = 1.50
        self.field_width = 1.30
        self.goal_width = 0.40
        self.goal_length = 0.10
        self.goal_pos = np.array([1.55, self.field_width / 2])

        self.robot_size = 0.08
        self.wheel_radius = 0.02
        self.wheel_separation = 0.08

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.field_length, self.field_width, self.field_length, self.field_width], dtype=np.float32)
        )

        pygame.init()
        self.screen_width = 1000
        self.screen_height = 800
        
        self.margin_x = 50
        self.margin_y = 50
        
        self.field_width_px = self.screen_width - 2 * self.margin_x
        self.field_height_px = self.screen_height - 2 * self.margin_y
        
        self.scale_x = self.field_width_px / self.field_length
        self.scale_y = self.field_height_px / self.field_width
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Simulação VSSS")

        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        self.max_steps = 1000
        self.current_step = 0

        self.robot_pos = np.array([0.1, 0.1])
        # Posição inicial aleatória para a bola
        self.ball_pos = np.array([
            np.random.uniform(0.1, self.field_length - 0.1),
            np.random.uniform(0.1, self.field_width - 0.1)
        ])
        
        self.robot_angle = 0.0
        
        # Adicionar variáveis para verificação de inatividade
        self.last_robot_pos = np.copy(self.robot_pos)
        self.inactivity_start_time = pygame.time.get_ticks()
        self.inactivity_threshold = 5000  # 5 segundos em milissegundos
        self.min_movement = 0.05  # Deslocamento mínimo em metros

    def reset(self):
        self.robot_pos = np.array([0.1, 0.1])
        # Posição inicial aleatória para a bola
        self.ball_pos = np.array([
            np.random.uniform(0.1, self.field_length - 0.1),
            np.random.uniform(0.1, self.field_width - 0.1)
        ])
        self.current_step = 0
        self.ball_velocity = np.array([0.0, 0.0])
        self.robot_angle = 0.0
        self.last_robot_pos = np.copy(self.robot_pos)
        self.inactivity_start_time = pygame.time.get_ticks()

        self.render()

        observation = np.array([self.robot_pos[0], self.robot_pos[1], self.ball_pos[0], self.ball_pos[1]], dtype=np.float32)       
        return observation, 0, False, False, {}

    def step(self, action):
        try:
            left_wheel_speed = action[0]
            right_wheel_speed = action[1]
            self._apply_motor_speeds(left_wheel_speed, right_wheel_speed)

            self.current_step += 1

            # Verificar inatividade
            current_time = pygame.time.get_ticks()
            distance_moved = np.linalg.norm(self.robot_pos - self.last_robot_pos)
            
            if distance_moved < self.min_movement:
                if current_time - self.inactivity_start_time > self.inactivity_threshold:
                    print("Robô inativo por mais de 5 segundos. Avançando para próxima iteração.")
                    return self.reset()
            else:
                self.inactivity_start_time = current_time
                self.last_robot_pos = np.copy(self.robot_pos)

            reward = self._calculate_reward(self.robot_pos, self.ball_pos)

            done = self._check_goal() or self.current_step >= self.max_steps

            if not done:
                done = self._check_inactivity()

            self.render()

            observation = np.array([self.robot_pos[0], self.robot_pos[1], 
                                  self.ball_pos[0], self.ball_pos[1]], dtype=np.float32)
            return observation, reward, done, False, {}
        
        except Exception as e:
            print(f"Erro durante a simulação: {str(e)}")
            return self.reset()
    
    def _check_goal(self):
        ball_radius = 0.025
        goal_width = self.goal_width
        goal_depth = 0.10

        # Verificar gol no gol esquerdo
        if self.ball_pos[0] <= ball_radius and abs(self.ball_pos[1] - self.field_width / 2) <= goal_width / 2:
            print("Gol no gol esquerdo!")
            return True

        # Verificar gol no gol direito
        if self.ball_pos[0] >= self.field_length - ball_radius and abs(self.ball_pos[1] - self.field_width / 2) <= goal_width / 2:
            print("Gol no gol direito!")
            return True

        return False
    
    def _check_inactivity(self):
        inactivity_threshold = 500  # Aumentado de 100 para 500
        min_movement = 0.01  # Aumentado de 0.0001 para 0.01

        if self.current_step > inactivity_threshold:
            distance_moved = np.linalg.norm(self.robot_pos - self._last_robot_pos)

            if distance_moved < min_movement:
                print("Robô parado por muito tempo. Avançando para a próxima iteração.")
                return True

        self._last_robot_pos = np.copy(self.robot_pos)
        return False

    def _apply_motor_speeds(self, left_speed, right_speed):
        # Manter uma velocidade base constante
        base_speed = 1.5  # Aumentado para movimento mais rápido
        linear_velocity = (left_speed + right_speed) * base_speed
        angular_velocity = (right_speed - left_speed) / self.wheel_separation

        ball_radius = 0.025
        robot_radius = self.robot_size / 2
        distance_to_ball = np.linalg.norm(self.robot_pos - self.ball_pos)

        # Calcular direção do gol
        ball_to_goal = self.goal_pos - self.ball_pos
        ball_to_goal = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-6)
        
        # Calcular posição alvo atrás da bola
        target_pos = self.ball_pos - ball_to_goal * (ball_radius + robot_radius)  # Reduzido de 0.1 para 0.05

        # Calcular direção do robô para o alvo
        direction_to_target = target_pos - self.robot_pos
        distance_to_target = np.linalg.norm(direction_to_target)

        if distance_to_target > 0:
            direction_to_target /= distance_to_target

        # Calcular direção atual do robô
        robot_direction = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        
        # Calcular o ângulo entre a direção do robô e a direção para o alvo
        dot_product = np.dot(robot_direction, direction_to_target)
        cross_product = np.cross(robot_direction, direction_to_target)
        angle_to_target = np.arctan2(cross_product, dot_product)
        angle_to_target = np.mod(angle_to_target + np.pi, 2 * np.pi) - np.pi

        # Ajustar velocidade baseada na distância e ângulo
        if distance_to_ball > ball_radius + robot_radius:
            # Se o ângulo for maior que 90 graus, inverter a direção
            if abs(angle_to_target) > np.pi / 2:
                linear_velocity *= -0.5
            # Reduzir velocidade quando estiver muito desalinhado
            angle_factor = np.cos(angle_to_target)
            linear_velocity *= max(0.6, angle_factor)  # Aumentado de 0.3 para 0.5
        else:
            # Quando estiver próximo da bola, reduzir a velocidade
            linear_velocity *= 0.8  # Aumentado de 0.5 para 0.8

        # Atualizar ângulo do robô com velocidade angular ajustada
        self.robot_angle += angular_velocity * 0.01 * 1.5  # Aumentado para rotação mais rápida

        # Movimento do robô
        new_x = self.robot_pos[0] + linear_velocity * np.cos(self.robot_angle) * 0.01
        new_y = self.robot_pos[1] + linear_velocity * np.sin(self.robot_angle) * 0.01

        # Limitar posição do robô
        new_x = np.clip(new_x, robot_radius, self.field_length - robot_radius)
        new_y = np.clip(new_y, robot_radius, self.field_width - robot_radius)

        new_robot_pos = np.array([new_x, new_y])
        distance = np.linalg.norm(new_robot_pos - self.ball_pos)

        # Verificar colisão
        if distance < (robot_radius + ball_radius):
            # Calcular vetor de separação
            separation_vector = (new_robot_pos - self.ball_pos) / (distance + 1e-6)
            
            # Ajustar posição do robô para evitar sobreposição
            new_robot_pos = self.ball_pos + separation_vector * (robot_radius + ball_radius)
            
            # Calcular velocidade do robô
            robot_velocity = np.array([
                linear_velocity * np.cos(self.robot_angle),
                linear_velocity * np.sin(self.robot_angle)
            ])
            
            # Transferir momento linear para a bola
            self.ball_velocity = robot_velocity * 0.9  # Aumentado para 0.9
            
            # Adicionar um pequeno impulso perpendicular
            perpendicular = np.array([-separation_vector[1], separation_vector[0]])
            self.ball_velocity += perpendicular * np.random.uniform(-0.01, 0.01)  # Reduzido para movimento mais estável
            
            # Manter a velocidade linear do robô após a colisão
            linear_velocity *= 0.95  # Reduzir apenas 5% da velocidade
            
            # Ajustar a velocidade da bola para seguir a direção do robô
            dot_product = np.dot(robot_velocity, separation_vector)
            if dot_product > 0:  # Se o robô está se movendo em direção à bola
                self.ball_velocity = robot_velocity * 0.95  # Transferir 95% da velocidade do robô

        self.robot_pos = new_robot_pos

        # Atualizar posição da bola
        new_ball_pos = self.ball_pos + self.ball_velocity * 0.01
        
        # Verificar colisão com as paredes
        if new_ball_pos[0] - ball_radius < 0:  # Parede esquerda
            new_ball_pos[0] = ball_radius
            self.ball_velocity[0] = -self.ball_velocity[0] * 0.9  # Reflexão com perda de energia
        elif new_ball_pos[0] + ball_radius > self.field_length:  # Parede direita
            new_ball_pos[0] = self.field_length - ball_radius
            self.ball_velocity[0] = -self.ball_velocity[0] * 0.9
            
        if new_ball_pos[1] - ball_radius < 0:  # Parede superior
            new_ball_pos[1] = ball_radius
            self.ball_velocity[1] = -self.ball_velocity[1] * 0.9
        elif new_ball_pos[1] + ball_radius > self.field_width:  # Parede inferior
            new_ball_pos[1] = self.field_width - ball_radius
            self.ball_velocity[1] = -self.ball_velocity[1] * 0.9
            
        self.ball_pos = new_ball_pos
        
        # Aplicar fricção
        self.ball_velocity *= self.friction
        
        # Limitar velocidade máxima da bola
        max_speed = 2.0
        current_speed = np.linalg.norm(self.ball_velocity)
        if current_speed > max_speed:
            self.ball_velocity *= max_speed / current_speed

    def _calculate_reward(self, robot_pos, ball_pos):
        # Distância à bola
        distance_to_ball = np.linalg.norm(robot_pos - ball_pos)
        
        # Distância ao gol
        distance_to_goal = np.linalg.norm(ball_pos - np.array([self.field_length, self.field_width / 2]))
        
        # Orientação do robô
        direction_to_ball = ball_pos - robot_pos
        direction_to_ball = direction_to_ball / (np.linalg.norm(direction_to_ball) + 1e-6)
        robot_direction = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        angle_error = np.arccos(np.clip(np.dot(direction_to_ball, robot_direction), -1.0, 1.0))
        
        # Velocidade da bola
        ball_speed = np.linalg.norm(self.ball_velocity)
        
        # Recompensa baseada em múltiplos fatores
        reward = (
            -1.0 * distance_to_ball +          # Prioridade para se aproximar da bola
            -0.5 * distance_to_goal +         # Incentivo para mover a bola em direção ao gol
            -0.3 * angle_error +              # Incentivo para alinhar com a bola
            -0.2 * ball_speed                 # Penalidade para movimentos muito bruscos
        )
        
        # Bônus para gol
        if self._check_goal():
            reward += 100.0
        
        return reward

    def render(self, mode='human'):
        self.screen.fill(self.WHITE)

        margin_x = 50
        margin_y = 50

        field_width_px = self.screen_width - 2 * margin_x
        field_height_px = self.screen_height - 2 * margin_y

        field_rect = pygame.Rect(margin_x, margin_y, field_width_px, field_height_px)
        pygame.draw.rect(self.screen, self.BLACK, field_rect, 2)

        mid_x = margin_x + field_width_px // 2
        pygame.draw.line(self.screen, self.BLACK, (mid_x, margin_y), (mid_x, margin_y + field_height_px), 2)

        goal_depth_px = int(0.10 * self.scale_x)
        goal_width_px = int(self.goal_width * self.scale_y)

        goal_left = pygame.Rect(
            margin_x - goal_depth_px,
            margin_y + (field_height_px - goal_width_px) // 2,
            goal_depth_px,
            goal_width_px
        )
        pygame.draw.rect(self.screen, self.BLACK, goal_left, 2)

        goal_right = pygame.Rect(
            margin_x + field_width_px,
            margin_y + (field_height_px - goal_width_px) // 2,
            goal_depth_px,
            goal_width_px
        )
        pygame.draw.rect(self.screen, self.BLACK, goal_right, 2)

        robot_pixel_pos = self._to_pixels(self.robot_pos)
        robot_size_px = int(self.robot_size * self.scale_x)
        robot_surface = pygame.Surface((robot_size_px, robot_size_px), pygame.SRCALPHA)
        pygame.draw.rect(robot_surface, self.BLUE, (0, 0, robot_size_px, robot_size_px))
        robot_rotated = pygame.transform.rotate(robot_surface, np.degrees(-self.robot_angle))
        robot_rect = robot_rotated.get_rect(center=self._to_pixels(self.robot_pos))
        self.screen.blit(robot_rotated, robot_rect.topleft)

        ball_pixel_pos = self._to_pixels(self.ball_pos)
        ball_radius_px = int(0.025 * self.scale_x)
        pygame.draw.circle(self.screen, self.RED, ball_pixel_pos, ball_radius_px)
        
        self.clock.tick(60)

        pygame.display.flip()

    def _to_pixels(self, pos):
        x = self.margin_x + int(pos[0] * self.scale_x)
        y = self.margin_y + int(pos[1] * self.scale_y)
        return x, y

    def close(self):
        pygame.quit()