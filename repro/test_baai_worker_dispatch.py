"""Regression tests for conductance addressing and output definitions."""
import sys
from pathlib import Path
import unittest
import json
import tempfile
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from _baai_worker_support import scale_conductance, select_readout

class Section:
    def __init__(self, mechanism, range_name, value):
        self.mechanism = mechanism
        self.segment = SimpleNamespace(**{mechanism: SimpleNamespace(**{range_name:value})})
    def has_membrane(self, name): return name == self.mechanism
    def __iter__(self): return iter([self.segment])

class DispatchTests(unittest.TestCase):
    def test_cached_rollout_cannot_be_reused_for_another_step(self):
        import _e42_recover_full_spectrum as recovery
        original = recovery.OUT
        try:
            with tempfile.TemporaryDirectory() as folder:
                recovery.OUT = Path(folder)
                responses = recovery.OUT / "responses"
                responses.mkdir()
                path = responses / "connection_0000_+1.npz"
                np.savez(path, voltage=np.ones((2, 3)))
                path.with_suffix(".json").write_text(json.dumps({"spec":{
                    "conn":0,"sign":1,"delta":.1,"tstop":2000.,"observable":"motor_voltage"}}))
                with self.assertRaises(ValueError):
                    recovery.run((0, 1))
        finally:
            recovery.OUT = original
    def test_compound_suffix_and_explicit_range_name(self):
        sections = [Section("slo1_egl19", "gbslo1", 2.), Section("slo1_unc2", "gbslo1", 3.)]
        self.assertEqual(scale_conductance(sections, "gbslo1", "slo1_egl19", .5), 1)
        self.assertEqual(sections[0].segment.slo1_egl19.gbslo1, 1.)
        self.assertEqual(sections[1].segment.slo1_unc2.gbslo1, 3.)
        scale_conductance(sections, "gbslo1_unc2", None, 2.)
        self.assertEqual(sections[1].segment.slo1_unc2.gbslo1, 6.)
    def test_missing_mechanism_cannot_become_zero_derivative(self):
        with self.assertRaises(ValueError):
            scale_conductance([Section("irk", "gbirk", 2.)], "gbslo1", "slo1_egl19", 2.)
    def test_readout_preserves_native_clipping_and_units(self):
        voltage = np.array([[-100., -40., 100.]])
        weights = np.ones((1,96))
        pre = select_readout(voltage, "muscle_preactivation", weights)
        activation = select_readout(voltage, "muscle_activation", weights)
        np.testing.assert_array_equal(pre[0], voltage[0])
        np.testing.assert_allclose(activation[0], [0., .4, .8])
        np.testing.assert_array_equal(select_readout(voltage, "motor_voltage"), voltage)
        with self.assertRaises(ValueError): select_readout(voltage, "muscle", weights)
        with self.assertRaises(ValueError): select_readout(voltage, "muscle_activation", weights.T)

if __name__ == "__main__": unittest.main()
