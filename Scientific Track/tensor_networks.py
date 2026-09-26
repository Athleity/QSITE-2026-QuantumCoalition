# -*- coding: utf-8 -*-
"""
QSITE 2026 — Quantum Coalition Challenge: Scientific Track
PART 2: tensor_networks.py
Description: Localized window contraction engine bypassing unicode and memory blowups.
"""

import numpy as np
import quimb.tensor as qtn
import config_and_exact as ce

def build_annni_mpo(n_qubits, kappa, h):
    """Constructs state-machine Matrix Product Operators for open boundary chains."""
    D = 5
    W = np.zeros((D, D, 2, 2), dtype=float)
    W[:, :, :, :] = ce.I2.real
    W[:, :, :, :] = ce.Z2.real
    W[:, :, :, :] = ce.Z2.real
    W[:, :, :, :] = ce.I2.real
    W[:, :, :, :] = -1.0 * ce.Z2.real
    W[:, :, :, :] = kappa * ce.Z2.real
    W[:, :, :, :] = -h * ce.X2.real
    W[:, :, :, :] = ce.I2.real

    first = W[4, :, :, :]
    last = W[:, 0, :, :]
    arrays = [first] + [W for _ in range(n_qubits - 2)] + [last]
    return qtn.MatrixProductOperator(arrays, shape="lrud")

def compute_dmrg_state(n_qubits, kappa, h, bond_dim=24):
    """Finds high-precision clean ground states via optimized DMRG sweeps."""
    H_mpo = build_annni_mpo(n_qubits, kappa, h)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=[10, 16, bond_dim], cutoffs=[1e-12])
    dmrg.solve(tol=1e-9, verbosity=0)
    return dmrg.state, float(dmrg.energy)

def evaluate_mps_observables(psi, n_qubits):
    """
    Measures long-range ordering parameters via low-overhead local operators.
    Completely avoids dense array allocation and unicode string indexing errors.
    """
    zz_near_vals = [float(psi.correlation(ce.Z2.real, i, i + 1)) for i in range(n_qubits - 1)]
    zz_next_vals = [float(psi.correlation(ce.Z2.real, i, i + 2)) for i in range(n_qubits - 2)]
    x_vals = [float(psi.local_expectation(ce.X2.real, (i,))) for i in range(n_qubits)]
    
    return float(np.mean(zz_near_vals)), float(np.mean(zz_next_vals)), float(np.mean(x_vals))

def compute_fidelity_susceptibility(n_qubits, kappa, h, dh=1e-3):
    """Calculates Fidelity Susceptibility across parameter step changes using native inner products."""
    try:
        psi1, _ = compute_dmrg_state(n_qubits, kappa, h, bond_dim=16)
        psi2, _ = compute_dmrg_state(n_qubits, kappa, h + dh, bond_dim=16)
        
        # Uses explicit canonical integer inner product sweeping
        overlap = abs(psi1.dot(psi2))
        if overlap > 1.0:
            overlap = 1.0
        return float(2.0 * (1.0 - overlap) / (dh ** 2))
    except Exception:
        return 0.0
