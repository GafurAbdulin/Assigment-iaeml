import jax
import chex
import equinox

from typing import Tuple

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState


class AutoResetWrapper(equinox.Module):
    env: BaseEnv = equinox.field(static=True)
    env_params: BaseEnvParams = equinox.field(static=True)
    init_state: BaseEnvState

    @equinox.filter_jit
    def reset(self, key: chex.PRNGKey, *args) -> Tuple[BaseEnvObservation, BaseEnvState]:
        return self.env.reset(key, self.env_params, self.init_state)

    @equinox.filter_jit
    def step(
        self, key: chex.PRNGKey, state: BaseEnvState, action: chex.Array, *args
    ) -> Tuple[BaseEnvObservation, BaseEnvState, chex.Scalar, chex.Array, dict]:
        """Reset after a terminal step while preserving its reward and info."""
        obs, new_state, reward, done, info = self.env.step(
            key, state, action, self.env_params
        )
        obs, new_state = jax.lax.cond(
            done, lambda: self.reset(key), lambda: (obs, new_state)
        )
        return obs, new_state, reward, done, info
