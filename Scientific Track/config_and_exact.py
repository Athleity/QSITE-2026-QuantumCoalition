# -*- coding: utf-8 -*-
"""
QSITE 2026 — Quantum Coalition Challenge: Scientific Track
PART 1: config_and_exact.py
Description: Configuration envelope constants and exact diagonalization simulation engine.
"""

import os
import numpy as np
from scipy.linalg import eigh

# ─────────────────────────────────────────────────────────────────
# Global Lattice Parameters
# ─────────────────────────────────────────────────────────────────
N_QUBITS_EXACT = 8     # Baseline small chain validation size
N_QUBITS_DMRG = 14     # Scaled scale for high-speed thermodynamic stability (Bypasses 8TB RAM error)
N_QUBITS_NOISY = 8     # Hardware VQE / Noise circuit scale

GRID_CLEAN = 16        # Grid scanning density for continuous contours
GRID_NOISY = 10        # Grid scanning density for noisy trajectories

VQE_LAYERS = 3         # Hardware ansatz layout depth layers
VQE_STEPS = 60         # Gradient steps
VQE_LR = 0.12          # Adam learning rate profile step

KAPPA_RANGE = (0.0, 1.0)
H_RANGE = (0.0, 2.0)
OUTPUT_DIR = "outputs"

# Base Matrices (Complex Arrays)
I2 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=complex)
Z2 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=complex)
X2 = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=complex)
Y2 = np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=complex)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _kron_single_site(op, site, n_qubits):
    """Generates structural identity Kronecker map matrices for a target site index."""
    res = op if site == 0 else I2
    for idx in range(1, n_qubits):
        next_op = op if idx == site else I2
        res = np.kron(res, next_op)
    return res

def _kron_two_site(op1, op2, site1, site2, n_qubits):
    """Generates structural identity Kronecker map matrices for paired site indices."""
    first = min(site1, site2)
    second = max(site1, site2)
    res = op1 if 0 == first else (op2 if 0 == second else I2)
    for idx in range(1, n_qubits):
        next_op = op1 if idx == first else (op2 if idx == second else I2)
        res = np.kron(res, next_op)
    return res

def build_dense_annni_hamiltonian(n_qubits, kappa, h, periodic=False):
    """Assembles full physical Hamiltonian matrices via precise array layout stacking."""
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits if periodic else n_qubits - 1):
        H -= _kron_two_site(Z2, Z2, i, (i + 1) % n_qubits, n_qubits)
    if kappa != 0.0:
        for i in range(n_qubits if periodic else n_qubits - 2):
            H += kappa * _kron_two_site(Z2, Z2, i, (i + 2) % n_qubits, n_qubits)
    if h != 0.0:
        for i in range(n_qubits):
            H -= h * _kron_single_site(X2, i, n_qubits)
    return np.real(H)

def compute_exact_ground_state(n_qubits, kappa, h, periodic=False):
    """Runs a standard full spectrum eigh solver to yield reference vectors."""
    H = build_dense_annni_hamiltonian(n_qubits, kappa, h, periodic=periodic)
    evals, evecs = eigh(H, eigvals=(0, 0))
    ground_energy = float(evals)
    psi = evecs[:, 0]
    
    zz_near = 0.0
    zz_next = 0.0
    x_mean = 0.0
    for i in range(n_qubits):
        if periodic or i < n_qubits - 1:
            m_near = _kron_two_site(Z2, Z2, i, (i + 1) % n_qubits, n_qubits)
            zz_near += np.vdot(psi, m_near @ psi).real
        if periodic or i < n_qubits - 2:
            m_next = _kron_two_site(Z2, Z2, i, (i + 2) % n_qubits, n_qubits)
            zz_next += np.vdot(psi, m_next @ psi).real
        m_x = _kron_single_site(X2, i, n_qubits)
        x_mean += np.vdot(psi, m_x @ psi).real
        
    return {
        "ground_energy": ground_energy,
        "psi": psi,
        "summary": {
            "zz_nearest_mean": zz_near / n_qubits,
            "zz_next_nearest_mean": zz_next / n_qubits,
            "x_mean": x_mean / n_qubits
        }
    }
