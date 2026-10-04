"""Meaningful coordinate/domain and retained-spectrum tests without flyvis."""
import unittest
import numpy as np
from flyvis_sensitivity_revision import chart_entry, coordinate_value, derivative, spectrum, comparison


class CoordinatesAndSpectra(unittest.TestCase):
    def test_log_chain_rule(self):
        e=chart_entry('nodes_time_const',0,.05,.05,1.)
        # f(theta)=theta^2; df/dz at z=0 is 2 theta^2.
        measured=derivative(e,1e-5,lambda z:np.array([coordinate_value(e,z)**2]),np.array([.05**2]))
        np.testing.assert_allclose(measured,[2*.05**2],rtol=1e-8)

    def test_zero_bias_is_not_removed(self):
        e=chart_entry('nodes_bias',0,0.,.5,1.)
        self.assertEqual(coordinate_value(e,-.2),-.2)
        measured=derivative(e,1e-5,lambda z:np.array([coordinate_value(e,z)]),np.array([0.]))
        np.testing.assert_allclose(measured,[1.])

    def test_boundary_is_right_derivative(self):
        e=chart_entry('edges_syn_strength',0,0.,.01,1.)
        with self.assertRaises(ValueError): coordinate_value(e,-.1)
        measured=derivative(e,.01,lambda z:np.array([3*coordinate_value(e,z)+coordinate_value(e,z)**2]),np.array([0.]))
        np.testing.assert_allclose(measured,[.03],rtol=1e-12)

    def test_full_spectrum_and_rank_ceiling(self):
        J=np.array([[3.,0.,0.,0.],[0.,1.,0.,0.]])
        out=spectrum(J)
        np.testing.assert_array_equal(out['eigenvalues'],[9.,1.,0.,0.])
        self.assertEqual(out['eff_dim_90'],1)
        self.assertEqual(out['eff_dim_99'],2)
        self.assertEqual(out['numeric_rank'],2)
        np.testing.assert_allclose(comparison(J,J)['relative_frobenius'],0.)

    def test_zero_sensitivity_has_zero_dimensions(self):
        out=spectrum(np.zeros((3,5)))
        self.assertEqual(out['eff_dim_90'],0)
        self.assertEqual(len(out['eigenvalues']),5)


if __name__=='__main__': unittest.main()
