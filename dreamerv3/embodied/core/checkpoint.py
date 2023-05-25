import concurrent.futures
import time

from . import basics
from . import path


class Checkpoint:

  def __init__(self, filename=None, log=True, parallel=True):
    self._filename = filename and path.Path(filename)
    self._log = log
    self._values = {}
    self._parallel = parallel
    if self._parallel:
      self._worker = concurrent.futures.ThreadPoolExecutor(1)
      self._promise = None

  def __setattr__(self, name, value):
    if name in ('exists', 'save', 'load'):
      return super().__setattr__(name, value)
    if name.startswith('_'):
      return super().__setattr__(name, value)
    has_load = hasattr(value, 'load') and callable(value.load)
    has_save = hasattr(value, 'save') and callable(value.save)
    if not (has_load and has_save):
      message = f"Checkpoint entry '{name}' must implement save() and load()."
      raise ValueError(message)
    self._values[name] = value

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self._values, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def latest_filename(self) -> path.Path:
    """Get latest checkpoint filename as a path.Path"""
    ckpt_paths = sorted(
      self._filename.parent.glob(self._filename.stem + "_[0-9]*.ckpt"),
      key=lambda p: int(p.stem.split('_')[-1])
    )
    if len(ckpt_paths) > 0:
      return ckpt_paths[-1]

    return self._filename

  @property
  def step(self) -> int:
    """Get current training step count"""
    assert "step" in self._values, self._values
    return int(self._values["step"])

  def exists(self, filename=None):
    assert self._filename or filename
    filename = path.Path(filename or self.latest_filename)
    exists = filename.exists()
    self._log and exists and print(f'Found existing checkpoint {filename}.')
    self._log and not exists and print('Did not find any checkpoint.')
    return exists

  def save(self, filename=None, keys=None):
    assert self._filename or filename
    save_filename = self._filename.parent / (self._filename.stem + f'_{self.step}.ckpt')
    filename = path.Path(filename or save_filename)
    filename.parent.mkdirs()
    self._log and print(f'Writing checkpoint: {filename}')
    if self._parallel:
      self._promise and self._promise.result()
      self._promise = self._worker.submit(self._save, filename, keys)
    else:
      self._save(filename, keys)

  def _save(self, filename, keys):
    keys = tuple(self._values.keys() if keys is None else keys)
    assert all([not k.startswith('_') for k in keys]), keys
    data = {k: self._values[k].save() for k in keys}
    data['_timestamp'] = time.time()
    if filename.exists():
      old = filename.parent / (filename.name + '.old')
      filename.copy(old)
      filename.write(basics.pack(data), mode='wb')
      old.remove()
    else:
      filename.write(basics.pack(data), mode='wb')
    self._log and print(f'Wrote checkpoint: {filename}')

  def load(self, filename=None, keys=None):
    assert self._filename or filename
    filename = path.Path(filename or self.latest_filename)
    self._log and print(f'Loading checkpoint: {filename}')
    data = basics.unpack(filename.read('rb'))
    keys = tuple(data.keys() if keys is None else keys)
    for key in keys:
      if key.startswith('_'):
        continue
      try:
        self._values[key].load(data[key])
      except Exception:
        print(f'Error loading {key} from checkpoint.')
        raise
    if self._log:
      age = time.time() - data['_timestamp']
      print(f'Loaded checkpoint from {age:.0f} seconds ago.')

  def load_or_save(self):
    if self.exists():
      self.load()
    else:
      self.save()
