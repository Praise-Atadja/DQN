import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import pygame


class PatientRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.window_size = 800
        self.max_steps = 100

        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(4,), dtype=int)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.obstacles = []
        self.goal = None
        self.current_reward = 0
        self.starting_point = np.array([0, 0])
        self.steps_taken = 0

    def _get_obs(self):
        return np.concatenate([self._agent_location, self._goal_location])

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._goal_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.starting_point.copy()
        self._goal_location = self.np_random.integers(
            0, self.size, size=2, dtype=int)
        while np.array_equal(self._goal_location, self._agent_location):
            self._goal_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)

        self.obstacles = [
            (np.array([0, 2]), 'Nurse'),
            (np.array([3, 2]), 'Doctor'),
            (np.array([3, 4]), 'Restricted'),
            (np.array([4, 1]), 'Emergency'),
            (np.array([1, 4]), 'Ward'),
            (np.array([1, 4]), 'Doctor'),
        ]

        self.current_reward = 0
        self.steps_taken = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location +
                               direction, 0, self.size - 1)

        reward = 0

        if new_location.tolist() not in [o[0].tolist() for o in self.obstacles]:
            self._agent_location = new_location
        else:
            self._agent_location = self.starting_point
            reward = -10

        terminated = np.array_equal(self._agent_location, self._goal_location)
        if terminated and self.steps_taken > 10:
            reward = 10
            self._agent_location = self._goal_location

        self.current_reward += reward
        self.steps_taken += 1

        done = terminated or self.steps_taken >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for obstacle, label in self.obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )
            font = pygame.font.SysFont(None, 24)
            text = font.render(label, True, (255, 255, 255))
            canvas.blit(text, (pix_square_size *
                        obstacle[0] + 5, pix_square_size * obstacle[1] + 5))

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        font = pygame.font.SysFont(None, 24)
        text = font.render('ECG Room', True, (255, 255, 255))
        canvas.blit(text, (pix_square_size *
                    self._goal_location[0] + 5, pix_square_size * self._goal_location[1] + 5))

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        font = pygame.font.SysFont(None, 16)
        text = font.render('Patient', True, (255, 255, 255))
        canvas.blit(text, ((self._agent_location + 0.5) * pix_square_size - (pix_square_size / 6),
                    (self._agent_location + 0.5) * pix_square_size - (pix_square_size / 6)))

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        font = pygame.font.SysFont(None, 20)
        reward_text = font.render(
            f'Reward: {self.current_reward}', True, (0, 0, 0))
        canvas.blit(reward_text, (5, self.window_size - 25))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id='gym_examples/GridWorld-v0',
    entry_point='patient_routing_env:PatientRoutingEnv',
    max_episode_steps=1000,
)
