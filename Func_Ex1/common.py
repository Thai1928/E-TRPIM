# Configuration, shared data structures, and dependencies
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
_GMSH_IMPORT_ERROR: Exception | None = None
_MESHIO_IMPORT_ERROR: Exception | None = None
try:
    import gmsh
except (ImportError, OSError) as exc:
    gmsh = None
    _GMSH_IMPORT_ERROR = exc
try:
    import meshio
except (ImportError, OSError) as exc:
    meshio = None
    _MESHIO_IMPORT_ERROR = exc
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False, 'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 120, 'savefig.dpi': 300})

@dataclass
class Config:
    case_name: str = 'ETRPIM_2D_PERFORATED_PLATE_COMPRESSIBLE_NEO_HOOKEAN'
    output_folder: str = 'OUTPUT_ETRPIM_Ex1'
    save_figures: bool = True
    colormap: str = 'jet'
    Lx: float = 1.0
    Ly: float = 2.0
    holes: List[Dict[str, float]] = field(default_factory=lambda: [{'cx': 0.25, 'cy': 0.5, 'R': 0.1}, {'cx': 0.75, 'cy': 1, 'R': 0.2}, {'cx': 0.5, 'cy': 1.5, 'R': 0.15}])
    E: float = 1000.0
    nu: float = 0.3
    thickness: float = 1.0
    prescribed_uy_top: float = 2.0
    lc_plate: float = 0.07
    lc_hole: float = 0.035
    regenerate_mesh: bool = True
    sample_n: int = 1000
    auto_sample_density: bool = True
    samples_per_node: float = 1
    sample_mode: str = 'halton'
    bottom_boundary_samples: int = 13
    top_boundary_samples: int = 13
    boundary_plot_samples: int = 160
    cheb_order: int = 1
    support_nodes: int = 18
    max_support_nodes: int = 40
    support_scale: float = 3.0
    cond_limit: float = 10000000000.0
    moment_regularization: float = 1e-12
    node_spacing_estimate: float = 0.05
    use_visibility_criterion: bool = True
    support_pool_factor: int = 6
    rbf_cover_pairwise_support: bool = True
    shape_value_limit: float = 20.0
    shape_gradient_limit_scaled: float = 100.0
    load_steps: int = 40
    iter_per_load_step: int = 1500
    lr: float = 0.001
    min_lr: float = 1e-07
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-08
    alpha_bottom: float = 1000000.0
    alpha_top: float = 1000000.0
    grad_clip_norm: float = 100000000.0
    max_coeff_update: float = 0.05
    reset_adam_each_load_step: bool = True
    print_every: int = 500
    min_J_allowed: float = 1e-08
    grad_tol: float = 0.0
    hard_impose_boundary_nodes: bool = True
    continuation_blend: float = 0.85
    max_backtracking: int = 14
    backtracking_factor: float = 0.5
    min_deformed_area_ratio: float = 0.01
    min_guard_J: float = 1e-05
    allow_relative_loss_increase: float = 1e-10
    fem_load_steps: int = load_steps
    fem_newton_max_iter: int = 25
    fem_newton_tol_res: float = 1e-06
    fem_newton_tol_du: float = 1e-08
    fem_fd_eps: float = 1e-07
    fem_n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    deformation_scale: float = 1.0
    field_figsize: Tuple[float, float] = (4.2, 6.8)
    component_disp_figsize: Tuple[float, float] = (13.0, 4.3)
    component_stress_figsize: Tuple[float, float] = (9.0, 7.8)
    profile_x: float = 0.75
    profile_n: int = 401
    profile_figsize: Tuple[float, float] = (12.0, 8.2)
    profile_markevery: int = 16

    def __getattr__(self, name):
        if name == 'mu':
            return self.E / (2.0 * (1.0 + self.nu))
        if name == 'lam':
            return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        if name == 'domain_area':
            return self.Lx * self.Ly - math.pi * sum((h['R'] ** 2 for h in self.holes))
        if name == 'base_support_radius':
            return self.support_scale * self.node_spacing_estimate
        if name == 'total_iterations':
            return self.load_steps * self.iter_per_load_step
        raise AttributeError(name)

@dataclass
class MeshData:
    nodes: np.ndarray
    elements: np.ndarray
    boundary_edges: Dict[str, np.ndarray]
    boundary_nodes: Dict[str, np.ndarray]

@dataclass
class ModelData:
    nodes: np.ndarray
    elements: np.ndarray
    fixed_bottom_nodes: np.ndarray
    top_nodes: np.ndarray
    x_int: np.ndarray
    w_int: np.ndarray
    x_bottom: np.ndarray
    w_bottom: np.ndarray
    x_top: np.ndarray
    w_top: np.ndarray
    x_guard: np.ndarray
    boundary_edges: Dict[str, np.ndarray]
    boundary_nodes: Dict[str, np.ndarray]

@dataclass
class Operators:
    Phi_int: sparse.csr_matrix
    DPhix_int: sparse.csr_matrix
    DPhiy_int: sparse.csr_matrix
    Phi_bottom: sparse.csr_matrix
    Phi_top: sparse.csr_matrix
    DPhix_top: sparse.csr_matrix
    DPhiy_top: sparse.csr_matrix
    DPhix_guard: sparse.csr_matrix
    DPhiy_guard: sparse.csr_matrix

@dataclass
class Fields:
    u: np.ndarray
    v: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    J: np.ndarray
    umag: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray
    vm: np.ndarray
    x_def: np.ndarray
