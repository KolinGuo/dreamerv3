import re

import embodied
import numpy as np


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  # Create checkpoint and load
  checkpoint = embodied.Checkpoint(logdir / 'models/model.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    max_single_reward = float(ep['reward'].max())
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
        'max_single_reward': max_single_reward,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        # Get last video index
        _key = f'{key}/video_'
        last_idx = max([-1] +
          [int(k.split(_key)[-1]) for k in metrics._lasts.keys() if _key in k])
        stats[f'{_key}{last_idx+1}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
      if re.match(args.log_keys_min, key):
        stats[f'min_{key}'] = ep[key].min(0).mean()
    metrics.add(stats, prefix=f'{mode}_ep_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)

  checkpoint.save()  # always save checkpoint instead of resume training
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), do_render=True)
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
