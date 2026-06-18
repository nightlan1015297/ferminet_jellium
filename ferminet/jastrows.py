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

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Union

import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  # e-e cusp augmented with jellium collective (CM + spin) dipole terms.
  MULTIPOLE = enum.auto()


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 0.25, params['ee_par'])
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], 0.5, params['ee_anti']))
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow():
  """Creates a Jastrow factor for electron-electron cusps."""

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      r_ee: jnp.ndarray,
      ae: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    del ae  # unused: the e-e cusp depends only on inter-electron distances.
    return _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun)

  return init, apply


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
  """Numerically stable softplus, used to keep coefficients non-negative."""
  return jnp.logaddexp(jnp.zeros_like(x), x)


def make_multipole_jastrow():
  r"""Creates an e-e-cusp Jastrow augmented with collective dipole terms.

  On top of the standard electron-electron cusp factor this adds two
  rotationally-invariant Gaussian factors built from collective coordinates of
  the electrons measured *relative to the reference centre* ``ae[:, 0]`` -- for
  the jellium sphere this is the charge-0 ghost atom at the sphere centre::

      J = J_ee_cusp
          - softplus(cm)          * | sum_i r_i     |^2   (center-of-mass dipole)
          - softplus(spin_dipole) * | sum_i s_i r_i |^2   (spin dipole)

  where ``r_i`` is electron ``i``'s position relative to the centre and
  ``s_i = +1`` for the first ``nspins[0]`` (spin-up) electrons and ``-1`` for the
  remaining ``nspins[1]`` (spin-down) electrons (matching the up-first electron
  ordering). ``softplus`` keeps both coefficients >= 0 so each factor is a
  normalisable Gaussian that *penalises* the corresponding collective
  fluctuation; a coefficient of 0 recovers the plain e-e-cusp Jastrow. The two
  coefficients are independent so each mode's contribution can be read off or
  frozen separately.

  Intended for the finite jellium sphere (a single, central reference atom and a
  rotationally symmetric confining background). For molecules the ``sum_i r_i``
  origin (atom 0) is arbitrary, so prefer ``SIMPLE_EE`` there.
  """

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying the electron-electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(shape=1)
    params['ee_anti'] = jnp.ones(shape=1)
    # Raw, pre-softplus coefficients. softplus(-4.) ~= 0.018: start near-off so
    # the collective terms switch on gently without disrupting the rest of the
    # ansatz (the coefficient is in absolute bohr^-2 units, so a small start is
    # safest across densities). Independent so each mode can be inspected/frozen
    # on its own; raise the init if you want them to engage faster.
    params['cm'] = jnp.full((1,), -4.0)
    params['spin_dipole'] = jnp.full((1,), -4.0)
    return params

  def apply(
      r_ee: jnp.ndarray,
      ae: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """e-e cusp plus center-of-mass and spin-dipole correlation."""
    cusp = _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun)

    # Electron positions relative to the reference centre (ghost atom 0).
    r = ae[:, 0, :]  # (nelectrons, ndim)
    # Collective dipole coordinates.
    p_cm = jnp.sum(r, axis=0)  # sum_i r_i                       (ndim,)
    signs = jnp.concatenate(
        [jnp.ones((nspins[0],)), -jnp.ones((nspins[1],))]
    )  # s_i = +1 (up) / -1 (down), aligned with the up-first ordering
    p_spin = jnp.sum(signs[:, None] * r, axis=0)  # sum_i s_i r_i (ndim,)

    a = _softplus(params['cm'][0])
    b = _softplus(params['spin_dipole'][0])
    multipole = -a * jnp.sum(p_cm**2) - b * jnp.sum(p_spin**2)
    return cusp + multipole

  return init, apply


def get_jastrow(jastrow: JastrowType):
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow()
  elif jastrow == JastrowType.MULTIPOLE:
    jastrow_init, jastrow_apply = make_multipole_jastrow()
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
