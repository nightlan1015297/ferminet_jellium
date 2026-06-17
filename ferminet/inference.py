# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Fixed-parameter VMC inference for FermiNet / jellium.

Runs an *inference-only* calculation: the network parameters are frozen
(restored from a trained checkpoint) and the script performs ``--steps``
Metropolis + local-energy-evaluation iterations, each over ``--batch_size``
walkers, accumulating a low-variance estimate of the total energy <H>. At the
end it prints

  * the number of trainable network parameters, and
  * the mean energy over the run, with a reblocked (autocorrelation-aware)
    statistical error bar.

This is the same machinery the README "Inference" section describes -- it
re-uses ``ferminet.train.train`` with ``optim.optimizer = 'none'`` (a pure
forward energy evaluation, no gradient/parameter update). The energy reported
each step by that path is the *unclipped* batch-mean local energy (see
``loss.make_loss.total_energy``), so averaging it over the run is an unbiased
VMC estimate of <H>.

Unlike a vanilla ``ferminet --config ... --config.optim.optimizer none`` re-run,
this script does *not* require the inference batch size or device count to match
the trained checkpoint. It restores only the trained *parameters* and
re-initialises the walkers at the requested batch size on the current devices,
then lets ``train.train`` re-equilibrate them with the configured MCMC burn-in
before accumulation. The trained run's outputs are never touched: results are
written to an ``inference_*`` subdirectory *inside* the restored run's directory.
The frozen network parameters are written once (in the staged checkpoint) and
deliberately omitted from the periodic checkpoints saved during the run, since
they never change.

Usage mirrors training -- pass the *same* system flags you trained with::

  python ferminet/inference.py \
      --config ferminet/configs/jellium_sphere.py \
      --config.system.n_electrons 18 --config.system.r_s 1

By default this runs 2**18 = 262144 steps of 4096 walkers each. Override with
``--steps`` and ``--batch_size``. Set the inference-only knobs through *this
script's* flags (``--steps`` / ``--batch_size`` / ``--burn_in``) rather than
``--config.optim.*`` / ``--config.batch_size`` so the trained run's recorded
config is left untouched.
"""

import csv
import math
import os

import numpy as np

# Hartree -> eV (CODATA), matching ferminet/configs/jellium_sphere.py.
HARTREE_TO_EV = 27.211386

# Defaults requested for the standard inference sweep.
DEFAULT_STEPS = 2 ** 18  # 262144
DEFAULT_BATCH_SIZE = 4096


# -----------------------------------------------------------------------------
# Energy statistics. Pure NumPy/stdlib so they can be unit-tested without the
# heavy JAX stack (which only needs to be present when actually running a chain).
# -----------------------------------------------------------------------------
def parse_energy(text):
  """Parses one ``energy`` cell from train_stats.csv into a real float.

  The cell is whatever ``str(np.asarray(loss))`` produced. For a real-valued
  wavefunction that is e.g. ``-2.345678``; for a complex wavefunction the local
  energy mean can carry a (physically negligible) imaginary part and prints as
  ``(-2.345678+0.0001j)``. The physical energy is the real part.

  Args:
    text: the raw CSV cell.

  Returns:
    The energy as a Python float (real part).

  Raises:
    ValueError: if the cell cannot be parsed as a real or complex number.
  """
  text = text.strip()
  try:
    return float(text)
  except ValueError:
    # complex() rejects the surrounding parens NumPy adds around 0-d complex
    # arrays, so strip them first: "(-2.3+0j)" -> "-2.3+0j".
    return complex(text.strip('()')).real


def reblock(values):
  """Flyvbjerg-Petersen reblocking of a (correlated) time series.

  Repeatedly averages adjacent pairs of samples; at each blocking level the
  standard error of the mean is recomputed. For correlated data the naive SEM
  (level 0) underestimates the true error; the estimate grows with blocking and
  plateaus once the block size exceeds the autocorrelation time.

  Args:
    values: 1-D sequence of per-step energies.

  Returns:
    List of ``(block_size, num_blocks, sem)`` tuples, one per blocking level,
    from the finest (block_size 1) upwards. Empty if fewer than 2 samples.
  """
  x = np.asarray(values, dtype=np.float64)
  levels = []
  block_size = 1
  while x.size >= 2:
    m = x.size
    sem = math.sqrt(x.var(ddof=1) / m)
    levels.append((block_size, m, sem))
    # Drop a trailing odd sample, then average adjacent pairs.
    if x.size % 2:
      x = x[:-1]
    x = 0.5 * (x[0::2] + x[1::2])
    block_size *= 2
  return levels


def reblocked_error(levels, min_blocks=8):
  """Picks a conservative error estimate from a reblocking curve.

  Takes the largest SEM among blocking levels that still retain at least
  ``min_blocks`` blocks. This captures the plateau while ignoring the noisy
  tail (very few, very large blocks), and never under-reports the naive SEM.

  Args:
    levels: output of :func:`reblock`.
    min_blocks: minimum number of blocks a level must have to be considered.

  Returns:
    The estimated standard error of the mean, or ``nan`` if no levels.
  """
  if not levels:
    return float('nan')
  candidates = [sem for (_, num_blocks, sem) in levels if num_blocks >= min_blocks]
  if not candidates:
    # Series too short to block meaningfully; fall back to the naive SEM.
    candidates = [levels[0][2]]
  return max(candidates)


def summarise_energy(csv_path, discard=0):
  """Reads a train_stats.csv and summarises the per-step energy column.

  Args:
    csv_path: path to the train_stats.csv written by the inference run.
    discard: number of leading steps to drop as extra equilibration (the
      configured MCMC burn-in already runs before any step is recorded, so 0 is
      usually appropriate).

  Returns:
    Dict with keys: n (samples used), mean, naive_sem, error (reblocked),
    tau (estimated integrated autocorrelation time, in steps), std, min, max.

  Raises:
    ValueError: if the file has no usable ``energy`` values.
  """
  energies = []
  with open(csv_path, newline='', encoding='UTF-8') as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames or 'energy' not in reader.fieldnames:
      raise ValueError(f'No "energy" column in {csv_path!r} (got '
                       f'{reader.fieldnames}).')
    for row in reader:
      cell = (row.get('energy') or '').strip()
      if not cell:
        continue
      try:
        energies.append(parse_energy(cell))
      except ValueError:
        continue  # skip the occasional malformed/NaN row rather than abort

  energies = np.asarray(energies, dtype=np.float64)
  if discard > 0:
    energies = energies[discard:]
  energies = energies[np.isfinite(energies)]
  n = int(energies.size)
  if n == 0:
    raise ValueError(f'No finite energies parsed from {csv_path!r}.')

  levels = reblock(energies)
  naive_sem = levels[0][2] if levels else float('nan')
  error = reblocked_error(levels)
  tau = (error / naive_sem) ** 2 if naive_sem and naive_sem > 0 else float('nan')
  return {
      'n': n,
      'mean': float(energies.mean()),
      'naive_sem': naive_sem,
      'error': error,
      'tau': tau,
      'std': float(energies.std(ddof=1)) if n > 1 else 0.0,
      'min': float(energies.min()),
      'max': float(energies.max()),
  }


# -----------------------------------------------------------------------------
# Checkpoint staging. Imports JAX/FermiNet lazily so the helpers above can be
# imported (and tested) in an environment without the full training stack.
# -----------------------------------------------------------------------------
def stage_inference_checkpoint(cfg, trained_ckpt, infer_dir):
  """Writes a fresh checkpoint that reuses trained params but new walkers.

  The trained checkpoint stores the walker configurations at the *training*
  batch size and device layout; restoring it directly would force inference to
  use that same batch size and device count. Instead we read just the trained
  ``params`` (and ``mcmc_width``) and pair them with freshly initialised walkers
  sized for the requested ``cfg.batch_size`` on the *current* device topology.

  The staged checkpoint is written with iteration ``t = -1`` so that, on
  restore, ``train.train`` sets the initial iteration to 0 -- which (a) triggers
  the MCMC burn-in that re-equilibrates the new walkers, and (b) runs the full
  ``cfg.optim.iterations`` steps. ``opt_state`` is a non-empty sentinel so the
  "Assuming inference run" branch fires regardless. On a later resume (after
  preemption) ``train.train`` instead finds the higher-numbered checkpoints this
  run saves and continues from them, skipping the burn-in -- the intended
  behaviour.

  Args:
    cfg: resolved config, already switched to inference settings.
    trained_ckpt: path to the trained checkpoint to take parameters from.
    infer_dir: directory to write the staged checkpoint into.

  Returns:
    The number of trainable parameters in the restored network.
  """
  # Lazy heavy imports -- only needed when actually staging/running a chain.
  import jax
  import jax.numpy as jnp
  from absl import logging
  from ferminet import checkpoint
  from ferminet import networks
  from ferminet import train

  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  if num_hosts != 1:
    raise ValueError(
        'inference.py stages walkers for a single host; found '
        f'{num_hosts} hosts. Run on one node (multiple GPUs are fine).')
  if cfg.batch_size % num_devices != 0:
    raise ValueError(
        f'batch_size ({cfg.batch_size}) must be divisible by the number of '
        f'devices ({num_devices}).')
  device_batch_size = cfg.batch_size // num_devices

  # Restore *only* the trained parameters (and the adapted MCMC width), bypassing
  # checkpoint.restore's device/batch-size checks on the stored walker data.
  logging.info('Reading trained parameters from %s', trained_ckpt)
  with open(trained_ckpt, 'rb') as f:
    raw = np.load(f, allow_pickle=True)
    params = raw['params'].tolist()
    try:
      mcmc_width = np.atleast_1d(np.asarray(raw['mcmc_width'].tolist()))
    except (KeyError, ValueError):
      mcmc_width = np.asarray([cfg.mcmc.move_width])

  # Trained checkpoints store params replicated across the *training* devices, so
  # each leaf carries a leading axis equal to the trained device count. Strip it
  # (take the device-0 copy) and re-replicate for the *current* device topology.
  # Without this, pmap on a different number of devices fails with an
  # IndivisibleError (e.g. params trained on 6 GPUs but inferring on 4: a leaf of
  # shape (6, ...) cannot be sharded across 4 devices). Re-replicating with the
  # current num_devices keeps the leading axis consistent with the fresh walkers.
  params = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], params)
  num_params = int(sum(np.size(x) for x in jax.tree_util.tree_leaves(params)))
  params = jax.tree_util.tree_map(
      lambda x: np.repeat(x[None], num_devices, axis=0), params)

  # Fresh walkers at the requested batch size, mirroring train.train's setup
  # (positions/spins from init_electrons; per-walker atoms/charges tiled).
  num_states = cfg.system.get('states', 0) or 1
  seed = 23 if cfg.debug.deterministic else int(np.random.SeedSequence().generate_state(1)[0])
  key = jax.random.PRNGKey(seed)
  pos, spins = train.init_electrons(
      key,
      cfg.system.molecule,
      cfg.system.electrons,
      batch_size=cfg.batch_size * num_states,
      init_width=cfg.mcmc.init_width,
      core_electrons={},
  )
  data_shape = (num_devices, device_batch_size)
  pos = np.asarray(jnp.reshape(pos, data_shape + (-1,)))
  spins = np.asarray(jnp.reshape(spins, data_shape + (-1,)))

  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  batch_atoms = np.asarray(
      jnp.tile(atoms[None, None], (num_devices, device_batch_size, 1, 1)))
  batch_charges = np.asarray(
      jnp.tile(charges[None, None], (num_devices, device_batch_size, 1)))

  data = networks.FermiNetData(
      positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges)

  os.makedirs(infer_dir, exist_ok=True)
  checkpoint.save(
      infer_dir,
      t=-1,                       # -> train.train starts at iteration 0 (burn-in runs)
      data=data,
      params=params,
      opt_state={'inference': True},  # non-None sentinel -> inference-run branch
      mcmc_width=mcmc_width,
  )
  return num_params


def _format_report(lines):
  bar = '=' * 64
  return '\n'.join([bar] + lines + [bar])


def main(argv):
  del argv
  from absl import flags
  from absl import logging
  from ferminet import base_config
  from ferminet import checkpoint

  flags_values = flags.FLAGS
  cfg = flags_values.config

  # Resolve the config exactly as training does (runs set_molecule, so for the
  # jellium config the system/molecule and save/restore paths are derived). At
  # this point optim/batch_size still hold the trained values, so this does not
  # change the trained run's recorded config.
  cfg = base_config.resolve(cfg)

  # Locate the trained checkpoint.
  train_dir = (flags_values.restore_path or cfg.log.restore_path
               or cfg.log.save_path)
  if not train_dir:
    raise SystemExit('Could not determine the trained checkpoint directory; '
                     'pass --restore_path.')
  trained_ckpt = checkpoint.find_last_checkpoint(train_dir)
  if not trained_ckpt:
    raise SystemExit(
        f'No trained checkpoint (qmcjax_ckpt_*.npz) found in {train_dir!r}. '
        'Train the system first, or point --restore_path at the trained run.')

  # Switch to inference settings *after* resolve.
  cfg.optim.optimizer = 'none'
  cfg.optim.iterations = flags_values.steps
  cfg.batch_size = flags_values.batch_size
  if flags_values.burn_in >= 0:
    cfg.mcmc.burn_in = flags_values.burn_in

  # Inference outputs go to a subdirectory *inside* the restored run's directory
  # so the results live alongside the checkpoint they came from, while the
  # trained run's own train_stats.csv / checkpoints (directly in train_dir) are
  # never overwritten.
  infer_dir = flags_values.save_path or os.path.join(
      train_dir.rstrip(os.sep),
      f'inference_b{cfg.batch_size}_s{cfg.optim.iterations}')
  if os.path.abspath(infer_dir) == os.path.abspath(train_dir):
    raise SystemExit('Refusing to write inference output into the trained run '
                     f'directory {train_dir!r}; choose a different --save_path.')
  cfg.log.save_path = infer_dir
  cfg.log.restore_path = infer_dir

  num_params = stage_inference_checkpoint(cfg, trained_ckpt, infer_dir)

  # Record what inference was run (next to its outputs).
  try:
    with open(os.path.join(infer_dir, 'inference_config.txt'), 'w',
              encoding='UTF-8') as f:
      f.write(repr(cfg.copy_and_resolve_references()))
  except (OSError, TypeError) as e:
    logging.info('Could not write inference_config.txt: %s', e)

  nelec = sum(cfg.system.electrons)
  logging.info('Inference: %s steps x %s walkers; network "%s" with %s params.',
               f'{cfg.optim.iterations:,}', f'{cfg.batch_size:,}',
               cfg.network.network_type, f'{num_params:,}')
  print(f'\nTrainable parameters: {num_params:,}\n'
        f'Running inference: {cfg.optim.iterations:,} steps x '
        f'{cfg.batch_size:,} walkers '
        f'(burn-in {cfg.mcmc.burn_in}, {cfg.mcmc.steps} MCMC steps/iter).\n'
        f'Output -> {infer_dir}\n', flush=True)

  # Run the inference chain (frozen params; writes train_stats.csv into
  # infer_dir). This is the long part -- intended for the GPU cluster.
  from ferminet import train
  train.train(cfg)

  # Summarise the recorded energies. Guarded so that a multi-hour run never
  # ends in a crash that hides the result: the per-step energies are on disk in
  # train_stats.csv and can always be re-summarised with summarise_energy.
  stats_csv = os.path.join(infer_dir, 'train_stats.csv')
  try:
    stats = summarise_energy(stats_csv, discard=flags_values.discard)
  except (OSError, ValueError) as err:
    print(f'\n[warning] Could not summarise {stats_csv}: {err}\n'
          f'Trainable parameters: {num_params:,}\n'
          f'The per-step energies are saved in {stats_csv}.', flush=True)
    return

  e = stats['mean']
  lines = [
      'FermiNet inference summary',
      '-' * 64,
      f'System            : N={nelec} electrons '
      f'({cfg.system.electrons[0]} up, {cfg.system.electrons[1]} down)',
      f'Network           : {cfg.network.network_type}, '
      f'{int(cfg.network.determinants)} determinants',
      f'Trainable params  : {num_params:,}',
      f'Restored params   : {trained_ckpt}',
      f'Inference output  : {infer_dir}',
      f'Steps averaged    : {stats["n"]:,}  (discarded {flags_values.discard})',
      f'Walkers per step  : {cfg.batch_size:,}',
      '-' * 64,
      f'Mean energy <H>   : {e:.6f} Ha',
      f'  reblocked error : +/- {stats["error"]:.6f} Ha '
      f'(naive SEM {stats["naive_sem"]:.6f}; autocorr tau ~ {stats["tau"]:.1f} steps)',
      f'  spread          : std {stats["std"]:.4f} Ha, '
      f'min {stats["min"]:.4f}, max {stats["max"]:.4f}',
  ]
  if nelec:
    e_per_e_ha = e / nelec
    e_per_e_ev = e * HARTREE_TO_EV / nelec
    lines.append(
        f'  energy/electron : {e_per_e_ha:.6f} Ha = {e_per_e_ev:.4f} eV')
  print('\n' + _format_report(lines), flush=True)


def main_wrapper():
  # Entry point. Flags are defined here (rather than at import time) so the
  # statistics helpers above remain importable without the absl/ml_collections
  # stack present.
  from absl import app
  from absl import flags
  from ml_collections.config_flags import config_flags

  config_flags.DEFINE_config_file('config', None, 'Path to config file.')
  flags.DEFINE_integer(
      'steps', DEFAULT_STEPS,
      'Number of inference iterations (energy evaluations). Default 2**18.')
  flags.DEFINE_integer(
      'batch_size', DEFAULT_BATCH_SIZE,
      'Number of MCMC walkers (samples) per step. Default 4096.')
  flags.DEFINE_integer(
      'burn_in', -1,
      'MCMC burn-in steps to re-equilibrate the freshly initialised walkers. '
      '-1 (default) keeps the value from the config.')
  flags.DEFINE_integer(
      'discard', 0,
      'Leading recorded steps to drop as extra equilibration when averaging. '
      'The configured burn-in already runs before step 0, so 0 is usual.')
  flags.DEFINE_string(
      'restore_path', '',
      'Directory of the trained run to take parameters from. Defaults to the '
      'config-derived path (cfg.log.restore_path / save_path).')
  flags.DEFINE_string(
      'save_path', '',
      'Directory for inference output. Defaults to '
      '"<trained-dir>/inference_b<batch>_s<steps>".')
  app.run(main)


if __name__ == '__main__':
  main_wrapper()
