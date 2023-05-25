import numpy as np

from . import base


class BatchEnv(base.Env):

  def __init__(self, envs, parallel, seed=None):
    assert all(len(env) == 0 for env in envs)
    assert len(envs) > 0
    self._envs = envs
    self._parallel = parallel
    self._keys = list(self.obs_space.keys())

    seed = np.random.randint(int(1e9)) if seed is None else seed
    self.seed(seed)

  @property
  def obs_space(self):
    return self._envs[0].obs_space

  @property
  def act_space(self):
    return self._envs[0].act_space

  def __len__(self):
    return len(self._envs)

  def step(self, action):
    assert all(len(v) == len(self._envs) for v in action.values()), (
        len(self._envs), {k: v.shape for k, v in action.items()})
    obs = []
    for i, env in enumerate(self._envs):
      act = {k: v[i] for k, v in action.items()}
      obs.append(env.step(act))
    if self._parallel:
      obs = [ob() for ob in obs]
    return {k: np.array([ob[k] for ob in obs]) for k in obs[0]}

  @property
  def info(self):
    if hasattr(self._envs[0], "info"):
      keys = self._envs[0].info
      return {k: np.array([env.info[k] for env in self._envs]) for k in keys}

    return None

  def render(self):
    return np.stack([env.render() for env in self._envs])

  def seed(self, seed):
    for i, env in enumerate(self._envs):
      env.seed(seed + i)
      [space.seed(seed + i) for space in env.act_space.values()]

  def close(self):
    for env in self._envs:
      try:
        env.close()
      except Exception:
        pass
