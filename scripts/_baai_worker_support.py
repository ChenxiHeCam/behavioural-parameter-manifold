"""Validated conductance addressing and motor-to-muscle output coordinates."""
import numpy as np

# Compound conductances belong to mechanism names with underscores; stripping
# the gb prefix does not give the mechanism name for these four variables.
MECHANISM = {
    "gbslo1_egl19": "slo1_egl19", "gbslo1_unc2": "slo1_unc2",
    "gbslo2_egl19": "slo2_egl19", "gbslo2_unc2": "slo2_unc2",
}


def scale_conductance(sections, channel, mechanism, factor):
    mechanism = mechanism or MECHANISM.get(channel, channel.removeprefix("gb"))
    range_name = channel.split("_", 1)[0] if channel in MECHANISM else channel
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Conductance multiplier must be positive and finite")
    touched = 0
    for sec in sections:
        if not sec.has_membrane(mechanism):
            continue
        for seg in sec:
            mech = getattr(seg, mechanism)
            value = getattr(mech, range_name)  # Fail on an invalid RANGE name.
            setattr(mech, range_name, value * factor)
            touched += 1
    if not touched:
        raise ValueError(f"Conductance {channel} in mechanism {mechanism} not found")
    return touched


def select_readout(voltage, observable, wout=None):
    voltage = np.asarray(voltage, dtype=np.float64)
    if voltage.ndim != 2:
        raise ValueError("Voltage must have shape (motor cells, time)")
    if observable == "motor_voltage":
        return voltage
    if observable == "muscle":
        raise ValueError("Ambiguous muscle readout: choose muscle_preactivation or muscle_activation")
    if observable not in ("muscle_preactivation", "muscle_activation"):
        raise ValueError(f"Unknown observable {observable}")
    weights = np.asarray(wout, dtype=np.float64)
    if weights.shape != (voltage.shape[0], 96):
        raise ValueError(f"Expected motor-to-muscle weights {(voltage.shape[0], 96)}, got {weights.shape}")
    preactivation = (voltage.T @ weights).T
    if observable == "muscle_preactivation":
        return preactivation
    # Matches WormNeuralNetwork.motor_neuron_volt_to_muscle_sig in BAAIWorm.
    return np.clip((preactivation + 80.0) / 100.0, 0.0, 0.8)
