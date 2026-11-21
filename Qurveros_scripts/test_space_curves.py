#!/usr/bin/env python3
"""
Generate NMR-style control fields (amplitude [kHz], phase [deg]) from four parametric space curves
using SCQC via the qurveros package.

Outputs (per curve):
  - pulses_type{1..4}.npy  (N x 2 array: [amplitude_kHz, phase_deg])
  - pulses_type{1..4}.csv  (same data with header)

Usage (defaults chosen so N ~ a few hundred):
  python make_controls_from_curves.py --N 400 --T_us 2000 --amax 25

Parameters:
  --N       Number of subpulses (rows) per pulse (default: 400)
  --T_us    Total pulse duration in microseconds (default: 2000 µs)
  --amax    Max RF amplitude in kHz after scaling (default: 25 kHz)
  --mode    Control mode for qurveros (default: XY; resonant control)

Notes:
  • qurveros returns resonant controls with Ω(t)=κ(t) (curvature) and Φ governed by torsion for 'XY' mode.
  • We convert Ω (angular freq) to “nutation frequency” units in kHz by dividing by 2π and scaling so max = amax.
  • Phases are converted to degrees and wrapped to [0, 360).
"""

import argparse
import numpy as np

# JAX / SciPy (JAX-scientific)
import jax
import jax.numpy as jnp
try:
    # Prefer JAX's special to keep things JIT-friendly
    from jax.scipy.special import ellipeinc
except Exception:
    # Fallback to SciPy if jax.scipy.special lacks ellipeinc in your environment
    from scipy.special import ellipeinc  # type: ignore

# qurveros
from qurveros.spacecurve import SpaceCurve

# --------------------------
# Curve definitions (in jnp)
# --------------------------

def curve_type_1(t, params=None):
    # t in [0, 4π]
    x = jnp.sin(t/2)
    y = jnp.sin(t) * jnp.cos(t)**2
    z = jnp.sin(t) * jnp.cos(t)
    return jnp.stack([x, y, z], axis=-1)

def curve_type_2(t, params=None):
    # t in [0, 2π]
    x = -0.5 * jnp.cos(t)**2
    y = 0.5 * (t - jnp.sin(t)*jnp.cos(t))
    z = jnp.sin(t)
    return jnp.stack([x, y, z], axis=-1)

def curve_type_3(t, params=None):
    # t in [0, 2π]
    # z uses incomplete elliptic integral of the second kind
    x = 0.25 * ((jnp.sqrt(2)/2.0) * jnp.sin(2*t) - 2*jnp.sin(t))
    y = 0.25 * ((jnp.sqrt(2)/2.0) * jnp.cos(2*t) + 2*jnp.cos(t))
    m = 2*jnp.sqrt(2)/(jnp.sqrt(2)+2.5)  # parameter m per your formula
    z = jnp.sqrt(jnp.sqrt(2)+2.5) * ellipeinc(1.5*t, m) / 3.0
    return jnp.stack([x, y, z], axis=-1)

def curve_type_4(t, params=None):
    # t in [0, 2π]
    x = (1.0/6.0) * (3*jnp.cos(t) - jnp.cos(3*t))
    y = (2.0/3.0) * jnp.sin(t)**3
    z = jnp.sin(t)
    return jnp.stack([x, y, z], axis=-1)

CURVE_SPECS = {
    1: (curve_type_1, (0.0, 4.0*np.pi)),
    2: (curve_type_2, (0.0, 2.0*np.pi)),
    3: (curve_type_3, (0.0, 2.0*np.pi)),
    4: (curve_type_4, (0.0, 2.0*np.pi)),
}

# -----------------------------------------
# Helper: build controls via qurveros (SCQC)
# -----------------------------------------

def get_controls_from_spacecurve(curve_fn, t_interval, N, control_mode='XY'):
    from qurveros.spacecurve import SpaceCurve

    sc = SpaceCurve(curve=curve_fn, order=0, interval=[t_interval[0], t_interval[1]], params=None)

    # Ask qurveros to sample exactly N points (no need to touch settings)
    sc.evaluate_frenet_dict(n_points=int(N))
    sc.evaluate_control_dict(control_mode, n_points=int(N))

    ctl = sc.control_dict
    if ('Omega' in ctl) and ('Phi' in ctl):
        Omega = np.asarray(ctl['Omega'])
        Phi   = np.asarray(ctl['Phi'])
        return Omega, Phi

    # Fallback if control_dict schema is different
    fd = sc.frenet_dict
    if ('kappa' in fd) and ('tau' in fd):
        kappa = np.asarray(fd['kappa'])
        tau   = np.asarray(fd['tau'])
    elif ('curvature' in fd) and ('torsion' in fd):
        kappa = np.asarray(fd['curvature'])
        tau   = np.asarray(fd['torsion'])
    else:
        raise RuntimeError("Could not locate Omega/Phi or curvature/torsion in qurveros outputs.")

    Omega = kappa
    Phi = np.concatenate([[0.0], np.cumsum(0.5*(tau[1:]+tau[:-1]))])
    if Phi.shape[0] != Omega.shape[0]:
        Phi = np.pad(Phi, (0, Omega.shape[0]-Phi.shape[0]), mode='edge')
    return Omega, Phi


# -----------------------------------------
# Conversion to NMR pulse table (kHz, deg)
# -----------------------------------------

def to_nmr_table(Omega, Phi, amax_kHz=25.0):
    """
    Convert angular frequency Ω (rad/s in arbitrary units) and phase Φ (rad) to:
        amplitudes in kHz (column 0), phases in degrees (column 1).
    We scale Ω/(2π) so that its max equals amax_kHz.
    """
    Omega = np.asarray(Omega).astype(float)
    Phi   = np.asarray(Phi).astype(float)

    # Convert angular frequency to cycles per second (Hz): divide by 2π
    Omega_cyc = Omega / (2.0*np.pi)

    # Scale so that max amplitude equals amax_kHz
    peak = np.max(np.abs(Omega_cyc))
    if peak <= 0:
        amp_kHz = np.zeros_like(Omega_cyc)
    else:
        scale = amax_kHz / (peak / 1e3)  # peak in Hz -> /1e3 to kHz
        amp_kHz = (Omega_cyc * scale) / 1e3  # convert Hz to kHz

    # Phase to degrees in [0, 360)
    phase_deg = (np.degrees(Phi) % 360.0 + 360.0) % 360.0

    return np.column_stack([amp_kHz, phase_deg])

# -----------------
# Main / CLI logic
# -----------------

def main():
    parser = argparse.ArgumentParser(description="Generate NMR-style control fields (kHz, deg) from four space curves via SCQC (qurveros).")
    parser.add_argument('--N', type=int, default=400, help='Number of subpulses (rows) per pulse [default: 400]')
    parser.add_argument('--T_us', type=float, default=2000.0, help='Total pulse duration in microseconds [default: 2000]')
    parser.add_argument('--amax', type=float, default=25.0, help='Max RF amplitude after scaling, in kHz [default: 25]')
    parser.add_argument('--mode', type=str, default='XY', help="qurveros control mode (e.g., 'XY' for resonant) [default: XY]")
    args = parser.parse_args()

    N = int(args.N)
    T_us = float(args.T_us)
    amax_kHz = float(args.amax)
    mode = str(args.mode)

    # Subpulse duration (uniform)
    dt_us = T_us / N

    for ctype in (1, 2, 3, 4):
        curve_fn, interval = CURVE_SPECS[ctype]

        # Build parameter grid in the SpaceCurve internally via qurveros options (N points)
        Omega, Phi = get_controls_from_spacecurve(curve_fn, interval, N, control_mode=mode)

        # Convert to NMR storage (kHz, deg)
        pulse_table = to_nmr_table(Omega, Phi, amax_kHz=amax_kHz)

        # Save
        np.save(f"pulses_type{ctype}.npy", pulse_table)
        header = f"Amplitude_kHz,Phase_deg  (N={N}, dt_us={dt_us:.6f}, total_T_us={T_us:.3f}, amax_kHz={amax_kHz:.3f}, mode={mode})"
        np.savetxt(f"pulses_type{ctype}.csv", pulse_table, delimiter=",", header=header, comments="")

        # Report
        print(f"[type {ctype}] Saved pulses_type{ctype}.npy / .csv  |  N={N}, dt={dt_us:.6f} µs, T={T_us:.3f} µs, amax={amax_kHz:.3f} kHz")

if __name__ == "__main__":
    main()
