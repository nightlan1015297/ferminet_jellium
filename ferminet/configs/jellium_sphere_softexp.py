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

r"""Closed-shell jellium sphere with the *softened-exponential* envelope.

Identical to ``ferminet/configs/jellium_sphere.py`` in every respect except the
multiplicative orbital envelope: this config uses

    exp(-sigma * sqrt(|r|^2 + a^2))     (ferminet.envelopes.make_softened_exponential_envelope)

instead of the default isotropic exponential exp(-sigma |r|). See that file for
the full documentation of every other knob and the reference energies.

Why the different envelope (jellium-specific):
  * The default exp(-sigma |r|) reproduces a nuclear (Kato) cusp at the centre.
    The jellium sphere has *no* point charge there -- inside the background the
    external potential is harmonic, V_ext(r) = const + (N/2 R_B^3) r^2 -- so the
    exact orbital is smooth (zero radial slope) at r = 0. The cusp injects a
    spurious 1/r term into the local energy at the centre, raising its variance
    and biasing the energy (worst at high density / small r_s, where the droplet
    is compact and electrons sit near the centre).
  * exp(-sigma sqrt(r^2 + a^2)) is smooth at r = 0 (Gaussian-like core, no cusp)
    yet decays exponentially in the tail (-> exp(-sigma r) as r -> inf), which
    matches the true asymptotics of a neutral cluster (the escaping electron
    sees a net +1 charge). The softening length ``a`` and rate ``sigma`` are
    learnable, one per orbital.

Run name / checkpoints: an envelope tag is appended to the auto run name so this
config writes to its own ``results/..._softened_exponential/`` directory and
never collides with (or restores from) the default isotropic-envelope runs.

  ferminet --config ferminet/configs/jellium_sphere_softexp.py \
      --config.system.n_electrons 2 \
      --config.system.r_s 1
"""

import os

from ferminet import base_config
from ferminet.jellium import hamiltonian as jellium_hamiltonian
from ferminet.utils import system


def set_jellium_sphere(cfg):
  """Derives the molecule, electrons and Hamiltonian kwargs from the system block.

  Run during ``base_config.resolve`` (after command-line parsing), so the user
  only needs to set the handful of physical knobs in ``cfg.system`` (number of
  electrons, ``r_s``, and -- optionally -- the background charge); the background
  radius, initial walker spread and local-energy kwargs follow automatically.

  Honours user overrides:
    * ``cfg.system.n_background``: <= 0 means "neutral" (background charge equals
      the electron count, the case studied in the paper). Set it to a positive
      value to study a charged sphere (background charge != electron count); this
      also rescales the background radius R_B and the constant self-energy.
    * ``cfg.mcmc.init_width``: <= 0 means "auto" (spread the initial walkers over
      the whole background sphere, R_B). Set a positive value to override.

  Args:
    cfg: ml_collections.ConfigDict after argument parsing.

  Returns:
    The updated ConfigDict.
  """
  n = int(cfg.system.n_electrons)
  r_s = float(cfg.system.r_s)
  if n <= 0 or n % 2 != 0:
    raise ValueError(
        'Closed-shell jellium spheres need an even, positive number of '
        f'electrons; got n_electrons={n}.')

  # Background charge. Default (<= 0) is a neutral sphere: N_bg == N_electrons.
  n_background = float(cfg.system.n_background)
  if n_background <= 0:
    n_background = float(n)

  # Background radius R_B = N_bg**(1/3) r_s (bohr) sets both V_ext and the
  # initial walker spread.
  r_b = jellium_hamiltonian.background_radius(n_background, r_s)

  with cfg.ignore_type():
    # A single charge-zero ghost atom at the origin defines the centre of the
    # background sphere and the one-electron coordinate system. Its charge (0) is
    # irrelevant to both the network (which only uses the atom *position*) and
    # the jellium Hamiltonian (which uses the background charge N_bg instead).
    cfg.system.molecule = [system.Atom(symbol='X', coords=(0.0, 0.0, 0.0))]
    # Closed shell: equal numbers of spin-up and spin-down electrons.
    cfg.system.electrons = (n // 2, n // 2)
    cfg.system.make_local_energy_kwargs = {
        'r_s': r_s,
        'n_background': n_background,
    }
    # Initialise the walkers spread over the whole background sphere rather than
    # bunched at the origin (init_electrons places them at the ghost atom).
    # init_width <= 0 means "auto"; a positive value is taken as given.
    if float(cfg.mcmc.init_width) <= 0:
      cfg.mcmc.init_width = float(r_b)

  # ----------------------------------------------------------------------------
  # Checkpoint paths. Derived here -- i.e. *after* the command line is parsed --
  # so they reflect any overridden settings rather than the file defaults.
  #
  #   * save_path    == 'auto' -> ./results/<run-name built from the resolved
  #                               system + ansatz>, so distinct settings never
  #                               clobber each other's checkpoints.
  #   * restore_path == 'auto' -> the same directory as save_path, so re-running
  #                               with identical knobs resumes in place. Pass an
  #                               explicit --config.log.restore_path to branch
  #                               from a different run instead.
  # Any explicit value (set in this file or on the command line) is left as-is.
  #
  # The envelope is tagged into the run name so this (softened-exponential) run
  # cannot collide with -- or restore from -- a default isotropic-envelope
  # checkpoint (whose orbital params have a different shape). An empty
  # make_envelope_fn (the default isotropic envelope) reproduces the original
  # jellium_sphere run name.
  # ----------------------------------------------------------------------------
  env_tag = ''
  if cfg.network.make_envelope_fn:
    env_name = cfg.network.make_envelope_fn.rsplit('.', maxsplit=1)[-1]
    env_tag = '_' + env_name.replace('make_', '').replace('_envelope', '')
  run_name = (
      f'jellium_N{n}_rs{r_s:g}_nbg{n_background:g}'
      f'_{cfg.network.network_type}_det{int(cfg.network.determinants)}{env_tag}'
  )
  if cfg.log.save_path == 'auto':
    cfg.log.save_path = os.path.join('results', run_name)
  if cfg.log.restore_path == 'auto':
    cfg.log.restore_path = cfg.log.save_path

  # Record exactly what was run, next to the checkpoints, so the output
  # directory is self-documenting (train.py does not persist the config itself).
  # Skipped when save_path is empty (i.e. a timestamped dir chosen later).
  if cfg.log.save_path:
    try:
      os.makedirs(cfg.log.save_path, exist_ok=True)
      with open(os.path.join(cfg.log.save_path, 'config.txt'), 'w') as f:
        f.write(repr(cfg.copy_and_resolve_references()))
    except (OSError, TypeError) as e:  # never let logging kill a training run
      print(f'[jellium_sphere_softexp] could not write config record: {e}')

  return cfg


def get_config():
  """Returns the config for a PsiFormer jellium-sphere calculation."""
  cfg = base_config.default()

  # ----------------------------------------------------------------------------
  # System (jellium-specific). Set N, r_s and optionally the background charge;
  # everything else (molecule, electrons, R_B, self-energy) is derived in
  # set_jellium_sphere after the command line is parsed.
  # ----------------------------------------------------------------------------
  cfg.system.n_electrons = 2       # Number of electrons. Even & positive.
  cfg.system.r_s = 4.0             # Wigner-Seitz density parameter (bohr).
  # Positive background charge +N_bg. <= 0 => neutral (N_bg = n_electrons), the
  # case in the paper. Set > 0 for a charged sphere.
  cfg.system.n_background = 0.0
  cfg.system.ndim = 3              # Must be 3 (the electrostatics are 3D-only).
  # Use the finite spherical-jellium Hamiltonian instead of the molecular one.
  cfg.system.make_local_energy_fn = 'ferminet.jellium.hamiltonian.local_energy'

  # No nuclei => no Hartree-Fock pretraining (PySCF cannot build this system).
  cfg.pretrain.method = None

  # ----------------------------------------------------------------------------
  # Network / ansatz. PsiFormer (von Glehn, Spencer, Pfau, ICLR 2023).
  #
  # Envelope: the *softened-exponential* envelope exp(-sigma sqrt(|r|^2 + a^2))
  # centred on the ghost atom -- smooth at the sphere centre (no spurious cusp,
  # since V_ext is harmonic there) with the correct exponential tail outside the
  # background. This is the only difference from jellium_sphere.py, which uses
  # the default isotropic exp(-sigma |r|). (Contrast the periodic HEG in
  # ferminet.pbc, which uses no decaying envelope at all.)
  # ----------------------------------------------------------------------------
  cfg.network.make_envelope_fn = (
      'ferminet.envelopes.make_softened_exponential_envelope')
  cfg.network.make_envelope_kwargs = {}
  cfg.network.network_type = 'psiformer'  # 'psiformer' or 'ferminet'.
  cfg.network.determinants = 8            # Number of determinants.
  cfg.network.full_det = True             # Dense (vs block-sparse) determinant.
  cfg.network.bias_orbitals = False       # Bias in the orbital output layer.
  cfg.network.jastrow = 'default'         # 'default', 'none', or 'simple_ee'.
  cfg.network.rescale_inputs = False      # Rescale inputs to grow as log(|r|).
  cfg.network.complex = True              # Real-valued wavefunction.
  # PsiFormer transformer hyperparameters (used when network_type='psiformer').
  cfg.network.psiformer.num_layers = 4
  cfg.network.psiformer.num_heads = 4
  cfg.network.psiformer.heads_dim = 64
  cfg.network.psiformer.mlp_hidden_dims = (256,)
  cfg.network.psiformer.use_layer_norm = True
  # FermiNet hidden dims (used when network_type='ferminet').
  cfg.network.ferminet.hidden_dims = ((256, 32), (256, 32), (256, 32), (256, 32))

  # ----------------------------------------------------------------------------
  # Optimisation / training. Modest defaults appropriate for the small (default
  # 2-electron) system; scale batch_size / iterations up for larger N.
  # ----------------------------------------------------------------------------
  cfg.batch_size = 3072            # MCMC walkers / batch size.
  cfg.optim.iterations = 150000     # Number of optimisation steps.
  cfg.optim.optimizer = 'kfac'     # 'kfac', 'adam', 'lamb', or 'none'.
  cfg.optim.laplacian = 'folx'  # 'default' or 'folx' (forward laplacian).
  cfg.optim.lr.rate = 0.05         # Learning rate.
  cfg.optim.lr.decay = 1.0         # Learning-rate decay exponent.
  cfg.optim.lr.delay = 10000.0     # Learning-rate decay scale.
  cfg.optim.clip_local_energy = 5.0  # Local-energy clipping window (in std).
  # KFAC knobs most worth tuning (full set in base_config).
  cfg.optim.kfac.damping = 0.001
  cfg.optim.kfac.norm_constraint = 0.001

  # ----------------------------------------------------------------------------
  # MCMC sampling.
  # ----------------------------------------------------------------------------
  cfg.mcmc.burn_in = 200           # Burn-in steps before optimisation.
  cfg.mcmc.steps = 20              # MCMC steps between network updates.
  # Width of the Gaussian used to place the initial walkers. <= 0 => auto
  # (spread over the whole background sphere, R_B); see set_jellium_sphere.
  cfg.mcmc.init_width = 0.0
  cfg.mcmc.move_width = 0.02       # Proposal width for random-walk Metropolis.
  cfg.mcmc.adapt_frequency = 100   # Steps between adapting the proposal width.
  cfg.mcmc.blocks = 1              # Number of blocks to split sampling into.

  # ----------------------------------------------------------------------------
  # Logging / checkpointing.
  # ----------------------------------------------------------------------------
  cfg.log.save_frequency = 30.0    # Minutes between checkpoints.
  cfg.log.stats_frequency = 1      # Iterations between stats logging.
  # 'auto' => derive from the resolved system in set_jellium_sphere (below).
  # Set an explicit path here or on the command line to override.
  cfg.log.save_path = 'auto'       # './results/<run-name>'; '' => timestamped dir.
  cfg.log.restore_path = 'auto'    # Same dir as save_path; '' => no restore.

  # ----------------------------------------------------------------------------
  # Observables (optional; off by default).
  # ----------------------------------------------------------------------------
  cfg.observables.s2 = False       # Total spin <S^2>.

  with cfg.ignore_type():
    cfg.system.set_molecule = set_jellium_sphere
    cfg.config_module = '.jellium_sphere_softexp'
  return cfg
