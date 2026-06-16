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

"""Tests for ferminet.jellium.hamiltonian."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import networks
from ferminet.configs import jellium_sphere
from ferminet.jellium import hamiltonian as jellium
import jax
import jax.numpy as jnp
import numpy as np


def gaussian_log_psi_signed(params, pos, spins=None, atoms=None, charges=None,
                            alpha=0.3):
  """A simple (non-antisymmetric) Gaussian test wavefunction.

  log|psi| = -alpha * sum_i |r_i|^2, for which the local kinetic energy is known
  analytically:  -1/2 nabla^2 ln|psi| - 1/2 |grad ln|psi||^2
                = 3 * alpha * N - 2 * alpha^2 * sum_i |r_i|^2.
  """
  del params, spins, atoms, charges
  log_psi = -alpha * jnp.sum(pos ** 2)
  return jnp.ones_like(log_psi), log_psi


class JelliumPotentialTest(parameterized.TestCase):
  """Checks the analytic building blocks against Sottile & Ballone (2001)."""

  @parameterized.parameters(
      (2, 1.0), (2, 4.0), (8, 3.25), (20, 5.62), (106, 2.0))
  def test_background_radius_and_self_energy(self, n, r_s):
    # R_B = N^(1/3) r_s  [Eq. (1)];  E_self = (3/5) N^(5/3) / r_s  [Eq. (4)].
    np.testing.assert_allclose(
        jellium.background_radius(n, r_s), n ** (1 / 3) * r_s, rtol=1e-12)
    np.testing.assert_allclose(
        jellium.background_self_energy(n, r_s),
        0.6 * n ** (5 / 3) / r_s, rtol=1e-12)

  @parameterized.parameters((2, 4.0), (8, 3.25), (20, 1.0))
  def test_vext_value_at_centre_and_edge(self, n, r_s):
    r_b = jellium.background_radius(n, r_s)
    # A single electron at the centre: V_ext(0) = -3N / (2 R_B).
    r_ae = jnp.array([[[0.0]]])
    np.testing.assert_allclose(
        jellium.potential_external(r_ae, n, r_s), -1.5 * n / r_b, rtol=1e-6)
    # A single electron exactly at the edge: V_ext(R_B) = -N / R_B (both
    # branches agree there -> V_ext is continuous).
    r_ae = jnp.array([[[r_b]]])
    np.testing.assert_allclose(
        jellium.potential_external(r_ae, n, r_s), -n / r_b, rtol=1e-6)

  def test_vext_continuous_at_boundary(self):
    n, r_s = 8, 3.25
    r_b = jellium.background_radius(n, r_s)
    eps = 1e-6
    v_in = jellium.potential_external(jnp.array([[[r_b - eps]]]), n, r_s)
    v_out = jellium.potential_external(jnp.array([[[r_b + eps]]]), n, r_s)
    np.testing.assert_allclose(v_in, v_out, atol=1e-4)

  def test_vext_matches_uniform_sphere_reference(self):
    n, r_s = 20, 2.0
    r_b = jellium.background_radius(n, r_s)
    r = np.array([0.1, 0.5 * r_b, r_b, 2.0 * r_b, 5.0 * r_b])
    inside = -0.5 * (n / r_b) * (3.0 - (r / r_b) ** 2)
    outside = -n / r
    expected = np.sum(np.where(r < r_b, inside, outside))
    r_ae = jnp.asarray(r)[:, None, None]
    np.testing.assert_allclose(
        jellium.potential_external(r_ae, n, r_s), expected, rtol=1e-6)


class JelliumLocalEnergyTest(parameterized.TestCase):

  @parameterized.parameters(['default', 'folx'])
  def test_local_energy_matches_analytic(self, laplacian):
    n, r_s, alpha = 2, 4.0, 0.3
    nspins = (1, 1)
    atoms = np.zeros((1, 3))  # background centred at the origin
    charges = np.zeros((1,))  # dummy ghost-atom charge

    log_psi = lambda p, x, s, a, c: gaussian_log_psi_signed(p, x, alpha=alpha)
    e_l = jellium.local_energy(
        log_psi, charges, nspins=nspins, laplacian_method=laplacian,
        r_s=r_s)

    np.random.seed(0)
    pos = np.random.normal(size=(64, n * 3)) * r_s
    keys = jax.random.split(jax.random.PRNGKey(0), num=pos.shape[0])
    batched = jax.vmap(
        e_l,
        in_axes=(None, 0,
                 networks.FermiNetData(
                     positions=0, spins=None, atoms=None, charges=None)))
    energies, _ = batched(
        {}, keys,
        networks.FermiNetData(positions=pos, spins=np.ones((n,)),
                              atoms=atoms, charges=charges))

    # Independent numpy reference for the same configurations.
    r_b = jellium.background_radius(n, r_s)
    e_self = jellium.background_self_energy(n, r_s)
    expected = []
    for cfg in pos:
      ri = cfg.reshape(n, 3)
      r = np.linalg.norm(ri, axis=1)
      kinetic = 3 * alpha * n - 2 * alpha ** 2 * np.sum(ri ** 2)
      v_ext = np.sum(np.where(r < r_b, -0.5 * (n / r_b) * (3 - (r / r_b) ** 2),
                              -n / np.where(r < r_b, r_b, r)))
      v_ee = 0.0
      for i in range(n):
        for j in range(i + 1, n):
          v_ee += 1.0 / np.linalg.norm(ri[i] - ri[j])
      expected.append(kinetic + v_ext + v_ee + e_self)
    np.testing.assert_allclose(energies, np.array(expected), rtol=1e-5)

  def test_raises_for_unsupported_options(self):
    charges = np.zeros((1,))
    log_psi = lambda p, x, s, a, c: gaussian_log_psi_signed(p, x)
    with self.assertRaises(NotImplementedError):
      jellium.local_energy(log_psi, charges, nspins=(1, 1), ndim=2, r_s=4.0)
    with self.assertRaises(NotImplementedError):
      jellium.local_energy(log_psi, charges, nspins=(1, 1), states=2, r_s=4.0)
    with self.assertRaises(NotImplementedError):
      jellium.local_energy(log_psi, charges, nspins=(1, 1),
                           pp_symbols=['X'], r_s=4.0)


class JelliumConfigTest(parameterized.TestCase):

  @parameterized.parameters((2, 4.0), (8, 3.25), (20, 1.0), (106, 5.62))
  def test_config_resolves(self, n, r_s):
    cfg = jellium_sphere.get_config()
    cfg.system.n_electrons = n
    cfg.system.r_s = r_s
    cfg = base_config.resolve(cfg)

    self.assertEqual(cfg.system.electrons, (n // 2, n // 2))
    self.assertLen(cfg.system.molecule, 1)
    self.assertEqual(cfg.system.molecule[0].symbol, 'X')
    self.assertEqual(cfg.system.molecule[0].charge, 0)
    self.assertEqual(
        cfg.system.make_local_energy_fn,
        'ferminet.jellium.hamiltonian.local_energy')
    self.assertAlmostEqual(cfg.system.make_local_energy_kwargs['r_s'], r_s)
    self.assertAlmostEqual(
        cfg.mcmc.init_width, n ** (1 / 3) * r_s, places=5)
    self.assertIsNone(cfg.pretrain.method)
    self.assertEqual(cfg.network.network_type, 'psiformer')

  def test_odd_electron_count_raises(self):
    cfg = jellium_sphere.get_config()
    cfg.system.n_electrons = 3
    with self.assertRaises(ValueError):
      base_config.resolve(cfg)


if __name__ == '__main__':
  absltest.main()
