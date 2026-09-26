# -*- coding: utf-8 -*-
"""
QSITE 2026 — Quantum Coalition Challenge: Scientific Track
PART 3: analysis.py
Description: Main grid loop runner, PennyLane circuit engine, and visualizer tools.
"""

import os
import sys
import time
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config_and_exact as ce
import tensor_networks as tn

def execute_mitigated_trajectory(params, n_qubits, p, kappa, h, seed=42):
    """Simulates a noisy quantum circuit and applies Zero-Noise Extrapolation (ZNE)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    rng = np.random.default_rng(seed)
    exact_res = ce.compute_exact_ground_state(n_qubits, kappa, h, periodic=False)
    state_vector = exact_res["psi"]
    
    def run_circuit(noise_rate):
        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state_vector, wires=range(n_qubits))
            for l in range(ce.VQE_LAYERS):
                for w in range(n_qubits):
                    qml.RY(params[l, w], wires=w)
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                    if noise_rate > 0.0 and rng.random() < noise_rate:
                        gate = rng.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
                        gate(wires=w + 1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        return float(circuit())

    val_p = run_circuit(p)
    val_2p = run_circuit(2.0 * p)
    return float(2.0 * val_p - val_2p)

def apply_publication_styling():
    """Applies high-contrast scientific plotting layouts to output windows."""
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#111111",
        "grid.color": "#EAEAEA",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True
    })

def generate_graphics_portfolio(kappas, hs, order_map, fidelity_map, noise_maps):
    """Generates all 2D contour lines and 3D mesh surface structures."""
    apply_publication_styling()
    KK, HH = np.meshgrid(kappas, hs)

    # Figure 1: 2D Contour System Plot
    fig1, ax1 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh1 = ax1.pcolormesh(KK, HH, order_map, cmap="RdBu_r", shading="gouraud", vmin=-1.0, vmax=1.0)
    cbar1 = fig1.colorbar(mesh1, ax=ax1, shrink=0.9)
    cbar1.set_label(r"Order Metric $\langle Z_i Z_{i+1} \rangle - |\langle Z_i Z_{i+2} \rangle|$", rotation=270, labelpad=20)
    ax1.contour(KK, HH, order_map, levels=[0.0], colors="#111111", linewidths=2.5)
    ax1.set_xlabel(r"Frustration Index ($\kappa$)", fontsize=13)
    ax1.set_ylabel(r"Transverse Field Amplitude ($h$)", fontsize=13)
    ax1.set_title(f"Thermodynamic Phase Transitions (DMRG N={ce.N_QUBITS_DMRG})", fontweight="bold")
    fig1.savefig(os.path.join(ce.OUTPUT_DIR, "01_publication_2d_contour.png"), dpi=300)
    plt.close(fig1)

    # Figure 2: 3D Surface parameter landscape topology
    fig2 = plt.figure(figsize=(9, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(KK, HH, order_map, cmap="coolwarm", lw=0, antialiased=True, rstride=1, cstride=1, alpha=0.85)
    ax2.contour(KK, HH, order_map, zdir='z', offset=-1.5, levels=[0.0], colors='black', linewidths=2.2)
    ax2.set_xlabel(r"$\kappa$", fontsize=13, labelpad=10)
    ax2.set_ylabel(r"$h$", fontsize=13, labelpad=10)
    ax2.set_zlabel("Order Value", fontsize=13, labelpad=8)
    ax2.set_zlim(-1.5, 1.0)
    ax2.view_init(elev=28, azim=-60)
    fig2.colorbar(surf2, ax=ax2, shrink=0.55, pad=0.1)
    ax2.set_title("3D Order Parameter Topological Surface", fontweight="bold")
    fig2.savefig(os.path.join(ce.OUTPUT_DIR, "02_order_parameter_surface_3d.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: 3D Fidelity Susceptibility Peaks
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(KK[:, :-1], HH[:, :-1], fidelity_map[:, :-1], cmap="inferno", lw=0, antialiased=True, rstride=1, cstride=1)
    ax3.set_xlabel(r"$\kappa$", fontsize=13, labelpad=10)
    ax3.set_ylabel(r"$h$", fontsize=13, labelpad=10)
    ax3.set_zlabel(r"$\chi_F$", fontsize=13, labelpad=8)
    ax3.view_init(elev=34, azim=-45)
    fig3.colorbar(surf3, ax=ax3, shrink=0.55, pad=0.1)
    ax3.set_title("3D Quantum Fidelity Susceptibility Critical Peaks", fontweight="bold")
    fig3.savefig(os.path.join(ce.OUTPUT_DIR, "03_fidelity_susceptibility_peaks_3d.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: Multi-Panel Mitigated Comparison Maps
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True)
    titles = [r"ZNE Mitigated Space ($p=0.01$)", r"ZNE Mitigated Space ($p=0.05$)"]
    for idx, ax in enumerate(axes4):
        mesh_n = ax.pcolormesh(KK, HH, noise_maps[idx], cmap="viridis", shading="gouraud")
        ax.contour(KK, HH, noise_maps[idx], levels=[0.0], colors="white", linewidths=1.8, linestyles="--")
        ax.set_xlabel(r"$\kappa$", fontsize=12)
        ax.set_title(titles[idx], fontweight="bold")
        fig4.colorbar(mesh_n, ax=ax, shrink=0.85)
    axes4.set_ylabel(r"$h$", fontsize=13)
    fig4.savefig(os.path.join(ce.OUTPUT_DIR, "04_zne_mitigated_noise_comparison.png"), dpi=300)
    plt.close(fig4)

def run_master_pipeline():
    print("🥇 Commencing World-Class Elite Hybrid Tensor Network Pipeline Loop...")
    start_time = time.time()
    kappas = np.linspace(*ce.KAPPA_RANGE, ce.GRID_CLEAN)
    hs = np.linspace(*ce.H_RANGE, ce.GRID_CLEAN)
    
    order_map = np.zeros((ce.GRID_CLEAN, ce.GRID_CLEAN))
    fidelity_map = np.zeros((ce.GRID_CLEAN, ce.GRID_CLEAN))
    
    print(f" -> Mapping clean phase landscape via DMRG scaling ($N={ce.N_QUBITS_DMRG}$)...")
    for hi, h in enumerate(hs):
        for ki, kappa in enumerate(kappas):
            psi, _ = tn.compute_dmrg_state(ce.N_QUBITS_DMRG, kappa, h, bond_dim=16)
            zz_near, zz_next, x_mean = tn.evaluate_mps_observables(psi, ce.N_QUBITS_DMRG)
            order_map[hi, ki] = zz_near - abs(zz_next)
            if hi < ce.GRID_CLEAN - 1:
                fidelity_map[hi, ki] = tn.compute_fidelity_susceptibility(ce.N_QUBITS_EXACT, kappa, h)
        print(f"     [DMRG Engine Progress] Completed Row Slice Line: h = {h:.2f}")

    noise_maps = []
    mock_vqe_params = np.ones((ce.VQE_LAYERS, ce.N_QUBITS_EXACT)) * 0.45
    for p in [0.01, 0.05]:
        print(f" -> Executing Richardson Zero-Noise Extrapolation loops for error rate p = {p}...")
        n_map = np.zeros((ce.GRID_CLEAN, ce.GRID_CLEAN))
        for hi, h in enumerate(hs):
            for ki, kappa in enumerate(kappas):
                mitigated_res = execute_mitigated_trajectory(mock_vqe_params, ce.N_QUBITS_EXACT, p, float(kappa), float(h))
                n_map[hi, ki] = mitigated_res - abs(mitigated_res)
        noise_maps.append(n_map)

    print("🎨 Generating and exporting publication graphics suite...")
    generate_graphics_portfolio(kappas, hs, order_map, fidelity_map, noise_maps)
    print(f"\n✨ Success! All assets exported safely to the '{ce.OUTPUT_DIR}/' directory.")
    print(f"Total processing runtime duration elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_master_pipeline()
