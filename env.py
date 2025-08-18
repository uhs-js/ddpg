import config
import math
import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class ContinuousCartPoleEnv(gym.Env[np.ndarray, int | np.ndarray]):
    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: str | None = None
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.min_action = -1.0
        self.max_action = 1.0

        self.gravity = config.GRAVITY
        self.masscart = config.MASSCART
        self.masspole = config.MASSPOLE
        self.total_mass = self.masspole + self.masscart
        self.length = config.LENGTH
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400

        self.vpy_canvas = None
        self.vpy_cart = None
        self.vpy_pole = None
        self.vpy_axle = None
        self.vpy_ground = None

        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * min(max(float(action), self.min_action), self.max_action)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        speed_penalty = 0.01 * (np.square(x_dot) + np.square(theta_dot))
        reward -= np.trunc(speed_penalty).astype(np.float32)

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05
        )
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        try:
            from vpython import canvas, box, cylinder, sphere, vector, rate, color
        except Exception as e:
            raise DependencyNotInstalled(
                'vpython is not installed, run `pip install vpython` to enable the VPython renderer.'
            ) from e

        if self.vpy_canvas is None:
            self.vpy_canvas = canvas(title=(self.spec.id if self.spec else "ContinuousCartPoleEnv VPython Renderer (DDPG)"), width=self.screen_width, height=self.screen_height, background=color.white)
            self._vpy_cart_width = 0.6
            self._vpy_cart_height = 0.3
            self._vpy_pole_len = 2 * self.length
            self._vpy_pole_radius = 0.05

            self.vpy_ground = box(pos=vector(0, -0.05, 0), size=vector(self.x_threshold * 4, 0.01, 0.1), color=color.white)

            self.vpy_cart = box(pos=vector(0, self._vpy_cart_height / 2.0, 0), size=vector(self._vpy_cart_width, self._vpy_cart_height, 0.2), color=color.black)

            self.vpy_pole = cylinder(pos=self.vpy_cart.pos + vector(0, self._vpy_cart_height / 2.0, 0), axis=vector(0, self._vpy_pole_len, 0), radius=self._vpy_pole_radius, color=color.orange)

            self.vpy_axle = sphere(pos=self.vpy_cart.pos + vector(0, self._vpy_cart_height / 2.0, 0), radius=self._vpy_pole_radius * 1.2, color=color.blue)

        if self.state is None:
            return None

        x, x_dot, theta, theta_dot = self.state

        self.vpy_cart.pos.x = x
        pivot = self.vpy_cart.pos + vector(0, self._vpy_cart_height / 2.0, 0)
        self.vpy_pole.pos = pivot
        self.vpy_pole.axis = vector(self._vpy_pole_len * math.sin(theta), self._vpy_pole_len * math.cos(theta), 0)
        self.vpy_axle.pos = pivot

        rate(self.metadata["render_fps"])

    def close(self):
        if self.vpy_canvas is not None:
            try:
                self.vpy_canvas.visible = False
            except Exception:
                pass
            self.vpy_canvas = None
            self.vpy_cart = None
            self.vpy_pole = None
            self.vpy_axle = None
            self.vpy_ground = None
        self.isopen = False
