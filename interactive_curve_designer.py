import os
import json
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from helix_algorithm import calculate_space_curve_variable_offset_tau as calculate_space_curve_helix
from rotation_matrix_algorithm import calculate_space_curve as calculate_space_curve_rotation
from personal_code.pulse_string_to_array import parse_pulse_sequence

#run with: streamlit run interactive_curve_designer.py
# ============================================================
# Backend: errors
# ============================================================

def calculate_errors(
    curve,
    normalize_second_by_isoperimetric=True,
    third_order_constant=1.0,
):
    curve = np.asarray(curve, dtype=float)

    dr = np.diff(curve, axis=0)
    r_mid = 0.5 * (curve[:-1] + curve[1:])

    ds = np.linalg.norm(dr, axis=1)
    L = np.sum(ds)

    if L == 0:
        return np.nan, np.nan, np.nan

    first_vec = curve[-1] - curve[0]

    area_vec = 0.5 * np.sum(
        np.cross(r_mid, dr),
        axis=0,
    )

    third_vec = np.sum(
        np.cross(r_mid, np.cross(r_mid, dr)),
        axis=0,
    )

    e1 = np.linalg.norm(first_vec) / L

    e2 = np.linalg.norm(area_vec) / L**2
    if normalize_second_by_isoperimetric:
        e2 *= 4 * np.pi

    e3 = np.linalg.norm(third_vec) / L**3
    e3 *= third_order_constant

    return e1, e2, e3


# ============================================================
# Backend: global override logic
# ============================================================

def resolve_fixed_paras(paras):
    paras = paras.copy()

    if paras["delta_phi_global"] is not None:
        paras["delta_phiAB"] = paras["delta_phi_global"]
        paras["delta_phiBC"] = paras["delta_phi_global"]

    if paras["B1_global"] is not None:
        paras["B1A"] = paras["B1_global"]
        paras["B1B"] = paras["B1_global"]
        paras["B1C"] = paras["B1_global"]

    if paras["tau_global"] is not None:
        paras["tauA"] = paras["tau_global"]
        paras["tauB"] = paras["tau_global"]
        paras["tauC"] = paras["tau_global"]

    if paras["omegaPR_global"] is not None:
        paras["omegaPRA"] = paras["omegaPR_global"]
        paras["omegaPRB"] = paras["omegaPR_global"]
        paras["omegaPRC"] = paras["omegaPR_global"]

    return paras

def apply_loop_links(paras, link_AB=False, link_BC=False, link_AC=False):
    """
    Link loop parameters B1, tau, omegaPR between A/B/C.

    If linked, the target loop copies from the first loop in the link.
    Example:
        link_BC=True makes C copy B.
    """

    paras = paras.copy()

    if link_AB:
        paras["B1B"] = paras["B1A"]
        paras["tauB"] = paras["tauA"]
        paras["omegaPRB"] = paras["omegaPRA"]

    if link_BC:
        paras["B1C"] = paras["B1B"]
        paras["tauC"] = paras["tauB"]
        paras["omegaPRC"] = paras["omegaPRB"]

    if link_AC:
        paras["B1C"] = paras["B1A"]
        paras["tauC"] = paras["tauA"]
        paras["omegaPRC"] = paras["omegaPRA"]

    return paras

def build_PR_string_from_paras(
    phi0A=0.0,
    omegaPRA=10.0,
    omegaPRB=10.0,
    omegaPRC=10.0,
    delta_phiAB=180.0,
    delta_phiBC=180.0,
    tauA=40.0,
    tauB=40.0,
    tauC=40.0,
    B1A=100.0,
    B1B=100.0,
    B1C=100.0,
    order="ABBACC",
):
    phi1A = phi0A + omegaPRA * tauA

    if order == "AABBCC":
        phi0B = phi0A + delta_phiAB
        phi0C = phi0B + delta_phiBC

    elif order == "ABBACC":
        phi0B = phi1A + delta_phiAB
        phi0C = phi0A + delta_phiBC

    elif order == "ABCCBA":
        phi0B = phi1A + delta_phiAB
        phi0C = phi0B + omegaPRB * tauB + delta_phiBC

    else:
        raise ValueError(f"Unknown order: {order}")

    phi1B = phi0B + omegaPRB * tauB
    phi1C = phi0C + omegaPRC * tauC

    ramp_segments = {
        "A_up":   dict(B1=B1A, phi0=phi0A, phi1=phi1A, tau=tauA),
        "A_down": dict(B1=B1A, phi0=phi1A, phi1=phi0A, tau=tauA),

        "B_up":   dict(B1=B1B, phi0=phi0B, phi1=phi1B, tau=tauB),
        "B_down": dict(B1=B1B, phi0=phi1B, phi1=phi0B, tau=tauB),

        "C_up":   dict(B1=B1C, phi0=phi0C, phi1=phi1C, tau=tauC),
        "C_down": dict(B1=B1C, phi0=phi1C, phi1=phi0C, tau=tauC),
    }

    segment_orders = {
        "AABBCC": ["A_up", "A_down", "B_up", "B_down", "C_up", "C_down"],
        "ABBACC": ["A_up", "B_up", "B_down", "A_down", "C_up", "C_down"],
        "ABCCBA": ["A_up", "B_up", "C_up", "C_down", "B_down", "A_down"],
    }

    segments = [ramp_segments[name] for name in segment_orders[order]]

    PS_string = "\n".join(
        f"PR({seg['B1']}, {seg['phi0']}, {seg['phi1']}, {seg['tau']});"
        for seg in segments
    )

    return PS_string, segments

def sync_linked_ui_fields(link_AB=False, link_BC=False, link_AC=False):
    """
    Synchronize Streamlit UI fields for linked loops.

    This updates the visible number_input values, not just the backend paras.
    """

    fields = ["B1", "tau", "omegaPR"]

    for field in fields:
        if link_AB:
            st.session_state[f"{field}B"] = st.session_state[f"{field}A"]

        if link_BC:
            st.session_state[f"{field}C"] = st.session_state[f"{field}B"]

        if link_AC:
            st.session_state[f"{field}C"] = st.session_state[f"{field}A"]

def initialize_ui_state():
    defaults = {
        "B1A": 100.0,
        "B1B": 100.0,
        "B1C": 100.0,

        "tauA": 50.0,
        "tauB": 50.0,
        "tauC": 50.0,

        "omegaPRA": 5.0,
        "omegaPRB": 5.0,
        "omegaPRC": 5.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================
# Backend: construct six phase-ramp curve
# ============================================================

def paras2curve(
    phi0A=0.0,
    omegaPRA=10.0,
    omegaPRB=10.0,
    omegaPRC=10.0,
    delta_phiAB=180.0,
    delta_phiBC=180.0,
    tauA=40.0,
    tauB=40.0,
    tauC=40.0,
    B1A=100.0,
    B1B=100.0,
    B1C=100.0,
    order="ABBACC",
    method="helix",
    helix_resolution=200,
):
    """
    Units:
        tau: microseconds
        phases: degrees
        omegaPR: degrees / microsecond
        B1: percent of 10 kHz
    """

    PS_string, segments = build_PR_string_from_paras(
        phi0A=phi0A,
        omegaPRA=omegaPRA,
        omegaPRB=omegaPRB,
        omegaPRC=omegaPRC,
        delta_phiAB=delta_phiAB,
        delta_phiBC=delta_phiBC,
        tauA=tauA,
        tauB=tauB,
        tauC=tauC,
        B1A=B1A,
        B1B=B1B,
        B1C=B1C,
        order=order,
    )

    if method == "rotation":

        PS = parse_pulse_sequence(PS_string, 0.5)

        return calculate_space_curve_rotation(
            PS[:, 0],
            PS[:, 1],
            0,
            10 * 1e3,
            0.5 * 1e-6,
        )

    elif method == "helix":
        B1_arr = np.array([seg["B1"] for seg in segments])

        # Important:
        # The helix algorithm uses phase differences between constant-helix
        # segments to rotate the binormal. Therefore we construct effective
        # phases so that their differences equal true phase jumps:
        #
        #   phase[k] - phase[k-1] = phi0_k - phi1_{k-1}
        #
        phase_arr = np.zeros(len(segments))
        phase_arr[0] = segments[0]["phi0"]

        for k in range(1, len(segments)):
            true_phase_jump = segments[k]["phi0"] - segments[k - 1]["phi1"]
            phase_arr[k] = phase_arr[k - 1] + true_phase_jump

        tau_arr = np.array([seg["tau"] * 1e-6 for seg in segments])

        omegaPR_arr_deg_per_us = np.array([
            (seg["phi1"] - seg["phi0"]) / seg["tau"]
            for seg in segments
        ])

        omegaPR_arr_Hz = omegaPR_arr_deg_per_us * 1e6 / 360

        # Sign convention may need flipping depending on your final convention.
        offset_arr = -omegaPR_arr_Hz

        return calculate_space_curve_helix(
            B1_amplitudes=B1_arr,
            phases=phase_arr,
            offset=offset_arr,
            max_Rabi_frequency=10 * 1e3,
            tau=tau_arr,
            resolution=helix_resolution,
        )

    else:
        raise ValueError("method must be 'helix' or 'rotation'")


# ============================================================
# Plotting
# ============================================================

def make_curve_figure(curve, title="Space Curve"):
    curve = np.asarray(curve)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=curve[:, 0],
            y=curve[:, 1],
            z=curve[:, 2],
            mode="lines",
            name="curve",
            line=dict(width=5),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[curve[0, 0]],
            y=[curve[0, 1]],
            z=[curve[0, 2]],
            mode="markers",
            name="start",
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[curve[-1, 0]],
            y=[curve[-1, 1]],
            z=[curve[-1, 2]],
            mode="markers",
            name="end",
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[curve[0, 0], curve[-1, 0]],
            y=[curve[0, 1], curve[-1, 1]],
            z=[curve[0, 2], curve[-1, 2]],
            mode="lines",
            name="closure vector",
            line=dict(width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        height=750,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# ============================================================
# Saving
# ============================================================

def save_current_candidate(output_root, paras, errors, PS_string):
    os.makedirs(output_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_root, f"candidate_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    data = {
        "paras": paras,
        "errors": {
            "first_order": errors[0],
            "second_order": errors[1],
            "third_order": errors[2],
        },
        "PS_string": PS_string,
    }

    with open(os.path.join(output_dir, "candidate.json"), "w") as f:
        json.dump(data, f, indent=4)

    return output_dir


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Six Phase-Ramp Curve Designer",
    layout="wide",
)

st.title("Six Phase-Ramp Curve Designer")

st.sidebar.header("Global settings")

method = st.sidebar.selectbox(
    "Calculation method",
    ["helix", "rotation"],
    index=0,
)

order = st.sidebar.selectbox(
    "Ramp order",
    ["ABBACC", "ABCCBA", "AABBCC"],
    index=0,
)

helix_resolution = st.sidebar.slider(
    "Helix resolution per ramp",
    min_value=20,
    max_value=1000,
    value=200,
    step=20,
)

st.sidebar.divider()
st.sidebar.header("Global overrides")

use_global_delta_phi = st.sidebar.checkbox("Use global delta_phi", value=False)
delta_phi_global = st.sidebar.number_input(
    "delta_phi global",
    value=180.0,
    step=1.0,
    disabled=not use_global_delta_phi,
)

use_global_B1 = st.sidebar.checkbox("Use global B1", value=False)
B1_global = st.sidebar.number_input(
    "B1 global",
    value=100.0,
    step=1.0,
    disabled=not use_global_B1,
)

use_global_tau = st.sidebar.checkbox("Use global tau", value=True)
tau_global = st.sidebar.number_input(
    "tau global [µs]",
    value=50.0,
    step=1.0,
    disabled=not use_global_tau,
)

use_global_omegaPR = st.sidebar.checkbox("Use global omegaPR", value=True)
omegaPR_global = st.sidebar.number_input(
    "omegaPR global [deg/µs]",
    value=5.0,
    step=0.1,
    disabled=not use_global_omegaPR,
)

st.sidebar.divider()
st.sidebar.header("Link loops")

link_AB = st.sidebar.checkbox("Link A → B", value=False)
link_BC = st.sidebar.checkbox("Link B → C", value=True)
link_AC = st.sidebar.checkbox("Link A → C", value=False)

initialize_ui_state()

sync_linked_ui_fields(
    link_AB=link_AB,
    link_BC=link_BC,
    link_AC=link_AC,
)

st.sidebar.divider()
st.sidebar.header("Save")

output_root = st.sidebar.text_input(
    "Output directory",
    value="/Users/leon/Desktop/Physik/interactive_curve_candidates",
)


left, right = st.columns([2.2, 1.0])

with right:
    st.header("Parameters")

    st.subheader("Phases")
    phi0A = st.number_input("phi0A [deg]", value=0.0, step=1.0)
    delta_phiAB = st.number_input("delta_phiAB [deg]", value=180.0, step=1.0)
    delta_phiBC = st.number_input("delta_phiBC [deg]", value=180.0, step=1.0)

    st.subheader("B1 amplitudes [%]")
    B1A = st.number_input("B1A", value=100.0, step=1.0, key="B1A")
    B1B = st.number_input("B1B", value=100.0, step=1.0, key="B1B")
    B1C = st.number_input("B1C", value=100.0, step=1.0, key="B1C")

    st.subheader("Durations [µs]")
    tauA = st.number_input("tauA", value=50.0, step=1.0, key="tauA")
    tauB = st.number_input("tauB", value=50.0, step=1.0, key="tauB")
    tauC = st.number_input("tauC", value=50.0, step=1.0, key="tauC")
    st.subheader("Phase-ramp rates [deg/µs]")
    omegaPRA = st.number_input("omegaPRA", value=5.0, step=0.1, key="omegaPRA")
    omegaPRB = st.number_input("omegaPRB", value=5.0, step=0.1, key="omegaPRB")
    omegaPRC = st.number_input("omegaPRC", value=5.0, step=0.1, key="omegaPRC")

fixed_paras = dict(
    phi0A=phi0A,

    delta_phiAB=delta_phiAB,
    delta_phiBC=delta_phiBC,

    B1A=B1A,
    B1B=B1B,
    B1C=B1C,

    tauA=tauA,
    tauB=tauB,
    tauC=tauC,

    omegaPRA=omegaPRA,
    omegaPRB=omegaPRB,
    omegaPRC=omegaPRC,

    delta_phi_global=delta_phi_global if use_global_delta_phi else None,
    B1_global=B1_global if use_global_B1 else None,
    tau_global=tau_global if use_global_tau else None,
    omegaPR_global=omegaPR_global if use_global_omegaPR else None,

    order=order,
)

resolved_paras = resolve_fixed_paras(fixed_paras)

resolved_paras = apply_loop_links(
    resolved_paras,
    link_AB=link_AB,
    link_BC=link_BC,
    link_AC=link_AC,
)

curve_kwargs = {
    key: value
    for key, value in resolved_paras.items()
    if key not in [
        "delta_phi_global",
        "B1_global",
        "tau_global",
        "omegaPR_global",
    ]
}

PS_string, _ = build_PR_string_from_paras(**curve_kwargs)

curve = paras2curve(
    **curve_kwargs,
    method=method,
    helix_resolution=helix_resolution,
)

e1, e2, e3 = calculate_errors(curve)

with left:
    st.subheader("Curve")

    fig = make_curve_figure(
        curve,
        title=f"{order}, method={method}",
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("First-order error", f"{e1:.3e}")
    c2.metric("Second-order error", f"{e2:.3e}")
    c3.metric("Third-order error", f"{e3:.3e}")

    st.subheader("Resolved parameters")
    st.json(resolved_paras)

    st.subheader("Pulse sequence PR notation")
    st.code(PS_string, language="text")

    if st.button("Save current candidate"):
        output_dir = save_current_candidate(
            output_root=output_root,
            paras=resolved_paras,
            errors=(e1, e2, e3),
            PS_string=PS_string,
        )

        fig.write_html(os.path.join(output_dir, "curve.html"))

        st.success(f"Saved to {output_dir}")