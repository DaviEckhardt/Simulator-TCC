import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VSSSEnv(gym.Env):
    def __init__(self):
        super(VSSSEnv, self).__init__()
        self.clock = pygame.time.Clock()
        self.ball_velocity = np.array([0.0, 0.0])
        self.friction = 0.98

        self.field_length = 1.50
        self.field_width = 1.30
        self.goal_width = 0.40
        self.goal_length = 0.10
        self.goal_pos = np.array([self.field_length, self.field_width / 2])

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
        self.ball_pos = np.array([self.field_length / 2, self.field_width / 2])
        
        self.robot_angle = 0.0

    def reset(self):
        self.robot_pos = np.array([0.1, 0.1])
        self.ball_pos = np.array([self.field_length / 2, self.field_width / 2])
        self.current_step = 0

        self.render()

        observation = np.array([self.robot_pos[0], self.robot_pos[1], self.ball_pos[0], self.ball_pos[1]], dtype=np.float32)
        return observation, {}

    def step(self, action):
        left_wheel_speed = action[0]
        right_wheel_speed = action[1]
        self._apply_motor_speeds(left_wheel_speed, right_wheel_speed)

        self.current_step += 1

        reward = self._calculate_reward(self.robot_pos, self.ball_pos)

        done = self._check_goal() or self.current_step >= self.max_steps

        if not done:
            done = self._check_inactivity()

        self.render()

        observation = np.array([self.robot_pos[0], self.robot_pos[1], self.ball_pos[0], self.ball_pos[1]], dtype=np.float32)
        return observation, reward, done, False, {}
    
    def _check_goal(self):
        ball_radius = 0.025
        goal_width = self.goal_width
        goal_depth = 0.10

        if self.ball_pos[0] <= ball_radius and abs(self.ball_pos[1] - self.field_width / 2) <= goal_width / 2:
            print("Gol no gol esquerdo!")
            return True

        if self.ball_pos[0] >= self.field_length - ball_radius and abs(self.ball_pos[1] - self.field_width / 2) <= goal_width / 2:
            print("Gol no gol direito!")
            return True

        return False
    
    def _check_inactivity(self):
        inactivity_threshold = 100
        min_movement = 0.0001

        if self.current_step > inactivity_threshold:
            distance_moved = np.linalg.norm(self.robot_pos - self._last_robot_pos)

            if distance_moved < min_movement:
                print("Robô parado por muito tempo. Avançando para a próxima iteração.")
                return True

        self._last_robot_pos = np.copy(self.robot_pos)
        return False

    def _apply_motor_speeds(self, left_speed, right_speed):
        linear_velocity = (left_speed + right_speed) / 2
        angular_velocity = (right_speed - left_speed) / self.wheel_separation

        ball_radius = 0.025
        distance_to_ball = np.linalg.norm(self.robot_pos - self.ball_pos)

        if distance_to_ball > ball_radius + self.robot_size / 2:
            target_pos = self.ball_pos
        else:
            target_pos = self.goal_pos

        direction_to_target = target_pos - self.robot_pos
        distance_to_target = np.linalg.norm(direction_to_target)

        if distance_to_target > 0:
            direction_to_target /= distance_to_target

        robot_direction = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        angle_to_target = np.arctan2(
            direction_to_target[1] * robot_direction[0] - direction_to_target[0] * robot_direction[1],
            direction_to_target[0] * robot_direction[0] + direction_to_target[1] * robot_direction[1]
        )

        if abs(angle_to_target) > np.pi / 2:
            linear_velocity *= -1

        self.robot_angle += angular_velocity * 0.01

        new_x = self.robot_pos[0] + linear_velocity * np.cos(self.robot_angle) * 0.01
        new_y = self.robot_pos[1] + linear_velocity * np.sin(self.robot_angle) * 0.01

        robot_radius = self.robot_size / 2
        new_x = np.clip(new_x, robot_radius, self.field_length - robot_radius)
        new_y = np.clip(new_y, robot_radius, self.field_width - robot_radius)

        self.robot_pos = np.array([new_x, new_y])

        distance = np.linalg.norm(self.robot_pos - self.ball_pos)

        direction = np.array([0.0, 0.0])

        if distance < (robot_radius + ball_radius):
            direction = (self.ball_pos - self.robot_pos) / (distance + 1e-6)

            if (self.ball_pos[0] <= ball_radius or self.ball_pos[0] >= self.field_length - ball_radius or
                self.ball_pos[1] <= ball_radius or self.ball_pos[1] >= self.field_width - ball_radius):
                new_x = self.robot_pos[0] - direction[0] * (robot_radius + ball_radius - distance)
                new_y = self.robot_pos[1] - direction[1] * (robot_radius + ball_radius - distance)

        self.robot_pos = np.array([new_x, new_y])

        if (self.ball_pos[0] <= ball_radius or self.ball_pos[0] >= self.field_length - ball_radius or
            self.ball_pos[1] <= ball_radius or self.ball_pos[1] >= self.field_width - ball_radius):
            impact_force = linear_velocity * 0.1
        else:
            impact_force = linear_velocity * 0.5

        self.ball_velocity += direction * impact_force

        self.ball_pos += self.ball_velocity * 0.01

        self.ball_velocity *= self.friction

        self.ball_pos[0] = np.clip(self.ball_pos[0], ball_radius, self.field_length - ball_radius)
        self.ball_pos[1] = np.clip(self.ball_pos[1], ball_radius, self.field_width - ball_radius)

    def _calculate_reward(self, robot_pos, ball_pos):
        distance_to_ball = np.linalg.norm(robot_pos - ball_pos)
        distance_to_goal = np.linalg.norm(ball_pos - np.array([self.field_length, self.field_width / 2]))
        reward = -distance_to_ball - distance_to_goal
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