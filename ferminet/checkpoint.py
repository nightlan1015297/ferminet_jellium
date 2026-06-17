# Copyright 2020 DeepMind Technologies Limited.
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

"""Super simple checkpoints using numpy."""

import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from ferminet import networks
from ferminet import observables
import jax
import jax.numpy as jnp
import numpy as np


def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  if ckpt_path and os.path.exists(ckpt_path):
    files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt_' in f]
    # Handle case where last checkpoint is corrupt/empty.
    for file in sorted(files, reverse=True):
      fname = os.path.join(ckpt_path, file)
      with open(fname, 'rb') as f:
        try:
          np.load(f, allow_pickle=True)
          return fname
        except (OSError, EOFError, zipfile.BadZipFile):
          logging.info('Error loading checkpoint %s. Trying next checkpoint...',
                       fname)
  return None


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'ferminet_{timestamp}')
  ckpt_save_path = save_path or default_save_path
  if ckpt_save_path and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path)
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def save(save_path: str,
         t: int,
         data: networks.FermiNetData,
         params,
         opt_state,
         mcmc_width,
         density_state: Optional[observables.DensityState] = None,
         save_params: bool = True) -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation
    save_params: whether to write the network parameters. Set False during
      fixed-parameter inference, where the (frozen) params never change and so
      need only be stored once (in the initial staged checkpoint); the periodic
      checkpoints then carry just the evolving walker/MCMC state. On restore,
      :func:`restore` recovers the params from the params-bearing checkpoint in
      the same directory.

  Returns:
    path to checkpoint file.
  """
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  logging.info('Saving checkpoint %s', ckpt_filename)
  with open(ckpt_filename, 'wb') as f:
    np.savez(
        f,
        t=t,
        data=dataclasses.asdict(data),
        params=(params if save_params else {}),
        opt_state=np.asarray(opt_state, dtype=object),
        mcmc_width=mcmc_width,
        density_state=(dataclasses.asdict(density_state)
                       if density_state else None))
  return ckpt_filename


def _restore_params(ckpt_dir: Optional[str]):
  """Recovers network params from a params-bearing checkpoint in a directory.

  Fixed-parameter inference omits the (frozen) params from its periodic
  checkpoints (see :func:`save`), so a checkpoint restored mid-run may not carry
  them. This scans ``ckpt_dir`` for the first checkpoint that does contain
  params. Checkpoints are scanned in ascending filename order, so the staged
  checkpoint written at iteration -1 (``qmcjax_ckpt_-00001.npz`` -- the '-' sorts
  before the positive iterations) is found ahead of the param-less ones.

  Args:
    ckpt_dir: directory to search for a params-bearing checkpoint.

  Returns:
    The params pytree, or None if no checkpoint with params is found.
  """
  if not ckpt_dir or not os.path.exists(ckpt_dir):
    return None
  files = sorted(f for f in os.listdir(ckpt_dir) if 'qmcjax_ckpt_' in f)
  for file in files:
    fname = os.path.join(ckpt_dir, file)
    try:
      with open(fname, 'rb') as f:
        params = np.load(f, allow_pickle=True)['params'].tolist()
    except (OSError, EOFError, zipfile.BadZipFile, KeyError):
      continue
    if params:
      return params
  return None


def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    batch_size: total batch size to be used. If present, check the data saved in
      the checkpoint is consistent with the batch size requested for the
      calculation.

  Returns:
    (t, data, params, opt_state, mcmc_width) tuple, where
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.
    density_state: optional state of the density matrix calculation

  Raises:
    ValueError: if the leading dimension of data does not match the number of
    devices (i.e. the number of devices being parallelised over has changed) or
    if the total batch size is not equal to the number of MCMC configurations in
    data.
  """
  logging.info('Loading checkpoint %s', restore_filename)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    data = networks.FermiNetData(**ckpt_data['data'].item())
    params = ckpt_data['params'].tolist()
    if not params:
      # Inference checkpoints omit the frozen params (see save()); recover them
      # from the staged params-bearing checkpoint in the same directory.
      params = _restore_params(os.path.dirname(restore_filename))
      if not params:
        raise ValueError(
            f'Checkpoint {restore_filename!r} contains no network parameters '
            'and none could be recovered from its directory.')
    opt_state = ckpt_data['opt_state'].tolist()
    mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())
    if ckpt_data['density_state']:
      density_state = observables.DensityState(
          **ckpt_data['density_state'].item())
    else:
      density_state = None
    if data.positions.shape[0] != jax.device_count():
      raise ValueError(
          'Incorrect number of devices found. Expected'
          f' {data.positions.shape[0]}, found {jax.device_count()}.'
      )
    if (
        batch_size
        and data.positions.shape[0] * data.positions.shape[1] != batch_size
    ):
      raise ValueError(
          f'Wrong batch size in loaded data. Expected {batch_size}, found '
          f'{data.positions.shape[0] * data.positions.shape[1]}.')
  return t, data, params, opt_state, mcmc_width, density_state
