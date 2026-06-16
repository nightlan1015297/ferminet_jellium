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

r"""Closed-shell jellium sphere with the PsiFormer (fully adjustable config).

Reproduces (at the VMC level, with a more expressive ansatz) the finite
spherical jellium model of

  F. Sottile and P. Ballone, Phys. Rev. B 64, 045105 (2001).

``N`` electrons sit in the field of a uniform positive background of charge
``+N_bg`` confined to a sphere of radius ``R_B = N_bg**(1/3) r_s``. See
``ferminet.jellium.hamiltonian`` for the Hamiltonian.

The default is the *simplest* system in the paper: ``N = 2`` electrons at the
metallic density ``r_s = 4`` with a neutral background. Every knob below can be
changed either by editing this file or on the command line, e.g.

  ferminet --config ferminet/configs/jellium_sphere.py \
      --config.system.n_electrons 8 \
      --config.system.r_s 3.25 \
      --config.batch_size 4096 \
      --config.optim.iterations 20000 \
      --config.network.determinants 8 \
      --config.optim.lr.rate 0.03

Closed-shell ("magic") sizes studied in the paper:
  N = 2, 8, 18, 20, 34, 40, 58, 92, 106
Densities studied:
  r_s = 1, 2, 3.25, 4, 5.62

The ``system`` block of options below is jellium-specific; every other block
(``network``, ``optim``, ``mcmc``, ``log``, ``observables`` ...) is the standard
FermiNet machinery and is documented in ``ferminet/base_config.py`` -- the
options surfaced here are the ones most worth tuning for this system. Any field
of the base config not listed below can still be overridden on the command line
(e.g. ``--config.optim.kfac.damping 0.0005``).

Reference total energies *per electron* (eV) for comparison. FermiNet performs
VMC with a far more flexible ansatz than the paper's Slater-Jastrow trial wave
function, so the converged energy should sit at or below E_VMC and approach (or
beat) the fixed-node E_DMC. Convert to the Hartree total energy reported by
FermiNet with  E_tot[Ha] = N * E_per_electron[eV] / 27.211386.

  N    r_s     E_VMC      E_DMC      E_HF       E_LDA
  2    1        5.0861     5.0810     5.5689     5.4841
  2    2       -0.7123    -0.7147    -0.2600    -0.5188
  2    3.25    -1.6703    -1.6716    -1.2616    -1.5803
  2    4       -1.7431    -1.7440    -1.3530    -1.6820   <- default
  2    5.62    -1.6482    -1.6485    -1.2966    -1.6230

For the default (N=2, r_s=4): E_DMC = -1.7440 eV/electron, i.e. a total energy of
2 * (-1.7440) / 27.211386 = -0.12818 Ha. FermiNet should converge to ~ this value
(or slightly lower, being variational with optimised nodes).
"""

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
  # Network / ansatz. PsiFormer (von Glehn, Spencer, Pfau, ICLR 2023). The
  # isotropic envelope exp(-sigma |r|) centred on the ghost atom gives the
  # correct decay of the confined droplet's orbitals, so no custom envelope /
  # feature layer is needed (contrast the periodic HEG in ferminet.pbc).
  # ----------------------------------------------------------------------------
  cfg.network.network_type = 'psiformer'  # 'psiformer' or 'ferminet'.
  cfg.network.determinants = 4            # Number of determinants.
  cfg.network.full_det = True             # Dense (vs block-sparse) determinant.
  cfg.network.bias_orbitals = False       # Bias in the orbital output layer.
  cfg.network.jastrow = 'default'         # 'default', 'none', or 'simple_ee'.
  cfg.network.rescale_inputs = False      # Rescale inputs to grow as log(|r|).
  cfg.network.complex = False             # Real-valued wavefunction.
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
  cfg.batch_size = 1024            # MCMC walkers / batch size.
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
  cfg.log.save_path = './results/'           # Checkpoint dir ('' => timestamped dir).
  cfg.log.restore_path = ''        # Checkpoint to restore from ('' => none).

  # ----------------------------------------------------------------------------
  # Observables (optional; off by default).
  # ----------------------------------------------------------------------------
  cfg.observables.s2 = False       # Total spin <S^2>.

  with cfg.ignore_type():
    cfg.system.set_molecule = set_jellium_sphere
    cfg.config_module = '.jellium_sphere'
  return cfg
