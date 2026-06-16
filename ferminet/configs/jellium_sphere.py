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

r"""Closed-shell jellium sphere with the PsiFormer.

Reproduces (at the VMC level, with a more expressive ansatz) the finite
spherical jellium model of

  F. Sottile and P. Ballone, Phys. Rev. B 64, 045105 (2001).

``N`` electrons sit in the field of a uniform positive background of charge
``+N`` confined to a sphere of radius ``R_B = N**(1/3) r_s``. See
``ferminet.jellium.hamiltonian`` for the Hamiltonian.

The default is the *simplest* system in the paper: ``N = 2`` electrons at the
metallic density ``r_s = 4``. Change the system on the command line, e.g.

  ferminet --config ferminet/configs/jellium_sphere.py \
      --config.system.n_electrons 8 --config.system.r_s 3.25

Closed-shell ("magic") sizes studied in the paper:
  N = 2, 8, 18, 20, 34, 40, 58, 92, 106
Densities studied:
  r_s = 1, 2, 3.25, 4, 5.62

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
  """Derives the molecule, electrons and Hamiltonian kwargs from N and r_s.

  Run during ``base_config.resolve`` (after command-line parsing), so the user
  only needs to set ``cfg.system.n_electrons`` and ``cfg.system.r_s``; the
  background radius, initial walker spread and local-energy kwargs follow.

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

  r_b = jellium_hamiltonian.background_radius(n, r_s)  # bohr

  with cfg.ignore_type():
    # A single charge-zero ghost atom at the origin defines the centre of the
    # background sphere and the one-electron coordinate system. Its charge (0) is
    # irrelevant to both the network (which only uses the atom *position*) and
    # the jellium Hamiltonian (which uses the background charge N instead).
    cfg.system.molecule = [system.Atom(symbol='X', coords=(0.0, 0.0, 0.0))]
    # Closed shell: equal numbers of spin-up and spin-down electrons.
    cfg.system.electrons = (n // 2, n // 2)
    cfg.system.make_local_energy_kwargs = {'r_s': r_s, 'n_background': float(n)}
    # Initialise the walkers spread over the whole background sphere rather than
    # bunched at the origin (init_electrons places them at the ghost atom).
    cfg.mcmc.init_width = float(r_b)
  return cfg


def get_config():
  """Returns the config for a PsiFormer jellium-sphere calculation."""
  cfg = base_config.default()

  # System: set N and r_s; everything else is derived in set_jellium_sphere.
  cfg.system.n_electrons = 2
  cfg.system.r_s = 4.0
  cfg.system.ndim = 3
  # Use the finite spherical-jellium Hamiltonian instead of the molecular one.
  cfg.system.make_local_energy_fn = 'ferminet.jellium.hamiltonian.local_energy'

  # No nuclei => no Hartree-Fock pretraining (PySCF cannot build this system).
  cfg.pretrain.method = None

  # PsiFormer ansatz (von Glehn, Spencer, Pfau, ICLR 2023). The isotropic
  # envelope exp(-sigma |r|) centred on the ghost atom gives the correct decay
  # of the confined droplet's orbitals, so no custom envelope/feature layer is
  # needed (contrast the periodic HEG in ferminet.pbc).
  cfg.network.network_type = 'psiformer'
  cfg.network.determinants = 16
  cfg.network.full_det = True

  # Modest training budget appropriate for a small (default 2-electron) system.
  cfg.batch_size = 1024
  cfg.optim.iterations = 10000

  with cfg.ignore_type():
    cfg.system.set_molecule = set_jellium_sphere
    cfg.config_module = '.jellium_sphere'
  return cfg
