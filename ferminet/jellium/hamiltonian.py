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

r"""Local energy of the finite spherical jellium model (jellium sphere).

Implements the Hamiltonian of

  F. Sottile and P. Ballone, "Fixed-node diffusion Monte Carlo computations for
  closed-shell jellium spheres", Phys. Rev. B 64, 045105 (2001),

in which ``N`` electrons move in the electrostatic potential of a structureless,
uniform positive background of total charge ``N`` (the system is globally
neutral) confined to a sphere of radius

  R_B = N**(1/3) * r_s,

where ``r_s`` is the usual electron-gas (Wigner-Seitz) parameter. Everything is
in Hartree atomic units. The Hamiltonian [their Eq. (2)] is

  H = -1/2 sum_i grad_i^2
      + sum_i V_ext(|r_i|)
      + 1/2 sum_{i != j} 1 / |r_i - r_j|
      + E_self,

with the external (electron-background) potential [their Eq. (3)]

  V_ext(r) = -1/2 (N / R_B) (3 - r^2 / R_B^2),   r <  R_B   (inside background)
           = -N / r,                             r >= R_B   (outside background)

and the constant background self-energy [their Eq. (4)]

  E_self = (3/5) N**(5/3) / r_s.

``V_ext`` is the potential energy of a single electron (charge -1) in the field
of a uniformly charged sphere of total charge ``+N`` and radius ``R_B``; it is
continuous and continuously differentiable at ``r = R_B``. ``E_self`` is the
self-energy of that positive background. It is a configuration-independent
constant: it shifts the total energy (and therefore matters when comparing with
the paper) but does not affect the wavefunction optimisation or sampling.

Unlike the homogeneous electron gas (see ``ferminet.pbc``), the jellium sphere is
a *finite* system in open boundary conditions, so the bare Coulomb interactions
above are used directly -- no Ewald summation is required. The background is
centred on the (single) reference position supplied in ``data.atoms``; the
recommended setup (see ``ferminet/configs/jellium_sphere.py``) places a single
charge-zero ghost atom at the origin so that ``r_ae[i, 0]`` is the distance of
electron ``i`` from the centre of the sphere.
"""

from typing import Optional, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
import jax.numpy as jnp


def background_radius(n_background: float, r_s: float) -> float:
  """Returns the background radius R_B = N**(1/3) r_s (in bohr)."""
  return n_background ** (1.0 / 3.0) * r_s


def background_self_energy(n_background: float, r_s: float) -> float:
  """Returns the background Coulomb self-energy E_self = (3/5) N**(5/3) / r_s."""
  return 0.6 * n_background ** (5.0 / 3.0) / r_s


def potential_external(
    r_ae: jnp.ndarray, n_background: float, r_s: float
) -> jnp.ndarray:
  """Returns the electron-background potential sum_i V_ext(|r_i|), Eq. (3).

  Args:
    r_ae: Shape (nelectrons, natoms, 1). r_ae[i, 0, 0] gives the distance between
      electron i and the centre of the background sphere (the single reference
      atom). Only the first atom is used.
    n_background: total positive background charge N (= number of electrons for a
      neutral system).
    r_s: electron-gas density parameter setting the background radius.
  """
  # Distance of each electron from the centre of the sphere (the reference atom).
  r = r_ae[..., 0, 0]  # (nelectrons,)
  r_b = background_radius(n_background, r_s)
  v_inside = -0.5 * (n_background / r_b) * (3.0 - (r / r_b) ** 2)
  # Guard the 1/r branch against r = 0: ``jnp.where`` evaluates both branches, so
  # the unused (inside) electrons must not feed a zero into the division.
  safe_r = jnp.where(r < r_b, r_b, r)
  v_outside = -n_background / safe_r
  return jnp.sum(jnp.where(r < r_b, v_inside, v_outside))


def potential_energy(
    r_ae: jnp.ndarray, r_ee: jnp.ndarray, n_background: float, r_s: float
) -> jnp.ndarray:
  """Returns the total potential energy of a jellium-sphere configuration.

  Args:
    r_ae: Shape (nelectrons, natoms, 1). Electron-centre distances.
    r_ee: Shape (nelectrons, nelectrons, :). r_ee[i, j, 0] gives the distance
      between electrons i and j.
    n_background: total positive background charge N.
    r_s: electron-gas density parameter.
  """
  return (
      hamiltonian.potential_electron_electron(r_ee)
      + potential_external(r_ae, n_background, r_s)
      + background_self_energy(n_background, r_s)
  )


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    ndim: int = 3,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Optional[Sequence[str]] = None,
    *,
    r_s: float,
    n_background: Optional[float] = None,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function for the finite spherical jellium model.

  The calling convention matches ``ferminet.hamiltonian.local_energy`` (so this
  function can be selected via ``cfg.system.make_local_energy_fn``); the
  jellium-specific parameters are passed as keyword-only arguments via
  ``cfg.system.make_local_energy_kwargs``.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Charges of the reference atoms. For jellium these
      are dummy (zero) charges; only the centre position(s) in ``data.atoms`` are
      used, and the physical background charge is ``n_background``.
    nspins: Number of electrons of each spin. The background charge defaults to
      the total electron count so the sphere is globally neutral.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    ndim: Number of spatial dimensions. Must be 3 -- the electrostatics of the
      uniform background (R_B, V_ext, E_self) are specific to three dimensions.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of 'default' or 'folx'.
    states: Number of excited states. Not implemented for jellium spheres.
    state_specific: Not implemented for jellium spheres.
    pp_type: Pseudopotential type. Not used (jellium has no pseudopotentials).
    pp_symbols: Pseudopotential symbols. Not used.
    r_s: electron-gas (Wigner-Seitz) parameter setting the background density and
      radius R_B = N**(1/3) r_s.
    n_background: total positive background charge N. Defaults to the total
      number of electrons (a neutral sphere), which is the case studied in the
      paper.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction at a single MCMC configuration in ``data``.
  """
  if ndim != 3:
    raise NotImplementedError(
        'The jellium-sphere electrostatics are only defined in 3D, got '
        f'ndim={ndim}.')
  if states > 0 or state_specific:
    raise NotImplementedError(
        'Excited states are not implemented for the jellium sphere.')
  if pp_symbols:
    raise NotImplementedError(
        'Pseudopotentials are not implemented for the jellium sphere.')
  del pp_type

  if n_background is None:
    n_background = float(sum(nspins))

  ke = hamiltonian.local_kinetic_energy(
      f,
      use_scan=use_scan,
      complex_output=complex_output,
      laplacian_method=laplacian_method,
  )

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total local energy = kinetic + potential."""
    del key  # unused
    _, _, r_ae, r_ee = networks.construct_input_features(
        data.positions, data.atoms, ndim)
    potential = potential_energy(r_ae, r_ee, n_background, r_s)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
