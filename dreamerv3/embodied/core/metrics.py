import collections
import warnings

import numpy as np


class Metrics:

  def __init__(self):
    self._scalars = collections.defaultdict(list)
    self._lasts = {}

  def scalar(self, key, value):
    self._scalars[key].append(value)

  def image(self, key, value):
    self._lasts[key].append(value)

  def video(self, key, value):
    self._lasts[key].append(value)

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self._lasts[key] = value
      else:
        self._scalars[key].append(value)

  def result(self, reset=True):
    result = {}
    result.update(self._lasts)
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self._scalars.items():
        if 'ep_stats/' in key:
          result[key+"_max"] = np.nanmax(values).astype(np.float64)
          result[key+"_mean"] = np.nanmean(values, dtype=np.float64)
          result[key+"_min"] = np.nanmin(values).astype(np.float64)
        else:
          result[key] = np.nanmean(values, dtype=np.float64)
    reset and self.reset()
    return result

  def reset(self):
    self._scalars.clear()
    self._lasts.clear()
