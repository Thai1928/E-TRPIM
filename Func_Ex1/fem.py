# FEM reference solution
from __future__ import annotations
from .common import *
from . import common

def solve_fem(operation, *args, **kwargs):
    if operation == 'neo_hookean_P_matrix':
        F = args[0] if args else kwargs['F']
        mu = args[1] if len(args) > 1 else kwargs['mu']
        lam = args[2] if len(args) > 2 else kwargs['lam']
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise FloatingPointError(f'J <= 0 trong phần tử FEM: J={J:.6e}')
        FinvT = np.linalg.inv(F).T
        logJ = math.log(J)
        P = mu * (F - FinvT) + lam * logJ * FinvT
        return (P, J)
        return
    if operation == 'cauchy_from_P_F':
        P = args[0] if args else kwargs['P']
        F = args[1] if len(args) > 1 else kwargs['F']
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise FloatingPointError(f'J <= 0 khi tính Cauchy: J={J:.6e}')
        return (P @ F.T / J, J)
        return
    if operation == 'precompute_one_element_reference':
        e = args[0] if args else kwargs['e']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        nodes = args[2] if len(args) > 2 else kwargs['nodes']
        conn = elements[e]
        X = nodes[conn]
        x1, y1 = X[0]
        x2, y2 = X[1]
        x3, y3 = X[2]
        two_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(two_area_signed) <= 1e-16:
            raise ValueError(f'Phần tử suy biến e={e}')
        area = 0.5 * abs(two_area_signed)
        beta = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
        gamma = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
        gradN = np.stack([beta, gamma], axis=1) / two_area_signed
        dofs: List[int] = []
        for node_id in conn.tolist():
            dofs.extend([2 * node_id, 2 * node_id + 1])
        return (conn, area, gradN, np.asarray(dofs, dtype=np.int64))
        return
    if operation == 'precompute_reference_data':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        n_jobs = args[2] if len(args) > 2 else kwargs['n_jobs']
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)((delayed(partial(solve_fem, 'precompute_one_element_reference'))(e, elements, nodes) for e in range(elements.shape[0])))
        conn_all = np.stack([r[0] for r in results], axis=0)
        area_all = np.asarray([r[1] for r in results], dtype=np.float64)
        gradN_all = np.stack([r[2] for r in results], axis=0)
        dofs_all = np.stack([r[3] for r in results], axis=0)
        return (conn_all, area_all, gradN_all, dofs_all)
        return
    if operation == 'element_internal_force':
        ue = args[0] if args else kwargs['ue']
        gradN = args[1] if len(args) > 1 else kwargs['gradN']
        area = args[2] if len(args) > 2 else kwargs['area']
        thickness = args[3] if len(args) > 3 else kwargs['thickness']
        mu = args[4] if len(args) > 4 else kwargs['mu']
        lam = args[5] if len(args) > 5 else kwargs['lam']
        u_nodes = ue.reshape(3, 2)
        grad_u = u_nodes.T @ gradN
        F = np.eye(2) + grad_u
        P, J = solve_fem('neo_hookean_P_matrix', F, mu, lam)
        fe = np.zeros(6, dtype=np.float64)
        for a in range(3):
            fa = thickness * area * (P @ gradN[a])
            fe[2 * a:2 * a + 2] = fa
        return (fe, F, J)
        return
    if operation == 'element_tangent_numeric':
        ue = args[0] if args else kwargs['ue']
        gradN = args[1] if len(args) > 1 else kwargs['gradN']
        area = args[2] if len(args) > 2 else kwargs['area']
        thickness = args[3] if len(args) > 3 else kwargs['thickness']
        mu = args[4] if len(args) > 4 else kwargs['mu']
        lam = args[5] if len(args) > 5 else kwargs['lam']
        fd_eps = args[6] if len(args) > 6 else kwargs['fd_eps']
        Ke = np.zeros((6, 6), dtype=np.float64)
        f0, F0, J0 = solve_fem('element_internal_force', ue, gradN, area, thickness, mu, lam)
        for j in range(6):
            du = np.zeros(6, dtype=np.float64)
            du[j] = fd_eps
            fp, _, _ = solve_fem('element_internal_force', ue + du, gradN, area, thickness, mu, lam)
            fm, _, _ = solve_fem('element_internal_force', ue - du, gradN, area, thickness, mu, lam)
            Ke[:, j] = (fp - fm) / (2.0 * fd_eps)
        Ke = 0.5 * (Ke + Ke.T)
        return (Ke, f0, F0, J0)
        return
    if operation == 'assemble_element_chunk':
        element_ids = args[0] if args else kwargs['element_ids']
        dofs_all = args[1] if len(args) > 1 else kwargs['dofs_all']
        gradN_all = args[2] if len(args) > 2 else kwargs['gradN_all']
        area_all = args[3] if len(args) > 3 else kwargs['area_all']
        u_flat = args[4] if len(args) > 4 else kwargs['u_flat']
        thickness = args[5] if len(args) > 5 else kwargs['thickness']
        mu = args[6] if len(args) > 6 else kwargs['mu']
        lam = args[7] if len(args) > 7 else kwargs['lam']
        fd_eps = args[8] if len(args) > 8 else kwargs['fd_eps']
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        fint_indices: List[np.ndarray] = []
        fint_values: List[np.ndarray] = []
        sigma_chunk: List[List[float]] = []
        vm_chunk: List[float] = []
        J_chunk: List[float] = []
        for e in element_ids:
            dofs = dofs_all[e]
            ue = u_flat[dofs]
            gradN = gradN_all[e]
            area = area_all[e]
            Ke, fe, F, J = solve_fem('element_tangent_numeric', ue, gradN, area, thickness, mu, lam, fd_eps)
            rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(Ke.ravel())
            fint_indices.append(dofs)
            fint_values.append(fe)
            P, _ = solve_fem('neo_hookean_P_matrix', F, mu, lam)
            sigma, J_sigma = solve_fem('cauchy_from_P_F', P, F)
            sxx = float(sigma[0, 0])
            syy = float(sigma[1, 1])
            sxy = float(0.5 * (sigma[0, 1] + sigma[1, 0]))
            vm = math.sqrt(max(0.0, sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2))
            sigma_chunk.append([sxx, syy, sxy])
            vm_chunk.append(vm)
            J_chunk.append(J_sigma)
        return (np.concatenate(rows) if rows else np.empty(0, dtype=np.int64), np.concatenate(cols) if cols else np.empty(0, dtype=np.int64), np.concatenate(vals) if vals else np.empty(0, dtype=np.float64), np.concatenate(fint_indices) if fint_indices else np.empty(0, dtype=np.int64), np.concatenate(fint_values) if fint_values else np.empty(0, dtype=np.float64), np.asarray(sigma_chunk, dtype=np.float64), np.asarray(vm_chunk, dtype=np.float64), np.asarray(J_chunk, dtype=np.float64))
        return
    if operation == 'assemble_global_fem_system':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        dofs_all = args[2] if len(args) > 2 else kwargs['dofs_all']
        gradN_all = args[3] if len(args) > 3 else kwargs['gradN_all']
        area_all = args[4] if len(args) > 4 else kwargs['area_all']
        u = args[5] if len(args) > 5 else kwargs['u']
        cfg = args[6] if len(args) > 6 else kwargs['cfg']
        ndof = nodes.shape[0] * 2
        ne = elements.shape[0]
        n_jobs = min(max(1, cfg.fem_n_jobs), ne)
        chunk_size = int(math.ceil(ne / n_jobs))
        chunks = [np.arange(i, min(i + chunk_size, ne), dtype=np.int64) for i in range(0, ne, chunk_size)]
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)((delayed(partial(solve_fem, 'assemble_element_chunk'))(chunk, dofs_all, gradN_all, area_all, u.reshape(-1), cfg.thickness, cfg.mu, cfg.lam, cfg.fem_fd_eps) for chunk in chunks))
        rows = np.concatenate([r[0] for r in results])
        cols = np.concatenate([r[1] for r in results])
        vals = np.concatenate([r[2] for r in results])
        K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        K.sum_duplicates()
        f_int = np.zeros(ndof, dtype=np.float64)
        for result in results:
            np.add.at(f_int, result[3], result[4])
        sigma = np.vstack([r[5] for r in results])
        vm = np.concatenate([r[6] for r in results])
        J = np.concatenate([r[7] for r in results])
        return (K, f_int, sigma, vm, J)
        return
    if operation == 'build_fem_boundary_conditions':
        nodes = args[0] if args else kwargs['nodes']
        boundary_nodes = args[1] if len(args) > 1 else kwargs['boundary_nodes']
        prescribed_uy_top = args[2] if len(args) > 2 else kwargs['prescribed_uy_top']
        n_nodes = nodes.shape[0]
        bc_ux = np.full(n_nodes, np.nan, dtype=np.float64)
        bc_uy = np.full(n_nodes, np.nan, dtype=np.float64)
        bc_ux[boundary_nodes['bottom']] = 0.0
        bc_uy[boundary_nodes['bottom']] = 0.0
        bc_uy[boundary_nodes['top']] = prescribed_uy_top
        fixed_dofs: List[int] = []
        fixed_values: List[float] = []
        for i in range(n_nodes):
            if not np.isnan(bc_ux[i]):
                fixed_dofs.append(2 * i)
                fixed_values.append(float(bc_ux[i]))
            if not np.isnan(bc_uy[i]):
                fixed_dofs.append(2 * i + 1)
                fixed_values.append(float(bc_uy[i]))
        return (bc_ux, bc_uy, np.asarray(fixed_dofs, dtype=np.int64), np.asarray(fixed_values, dtype=np.float64))
        return
    if operation == 'solve_fem_reference':
        mesh = args[0] if args else kwargs['mesh']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        print('\nĐang tiền xử lý dữ liệu phần tử FEM ...')
        _, area_all, gradN_all, dofs_all = solve_fem('precompute_reference_data', mesh.nodes, mesh.elements, cfg.fem_n_jobs)
        _, _, fixed_dofs, fixed_values = solve_fem('build_fem_boundary_conditions', mesh.nodes, mesh.boundary_nodes, cfg.prescribed_uy_top)
        ndof = mesh.nodes.shape[0] * 2
        all_dofs = np.arange(ndof, dtype=np.int64)
        free_mask = np.ones(ndof, dtype=bool)
        free_mask[fixed_dofs] = False
        free_dofs = all_dofs[free_mask]
        u = np.zeros(ndof, dtype=np.float64)
        step_logs: List[Dict[str, float]] = []
        t0 = time.perf_counter()
        print('\n' + '=' * 78)
        print('FEM T3 Neo-Hookean nén được - Newton-Raphson tham chiếu')
        print(f'Số nút={mesh.nodes.shape[0]}, số phần tử={mesh.elements.shape[0]}')
        print(f'E={cfg.E:g}, nu={cfg.nu:g}, mu={cfg.mu:g}, lambda={cfg.lam:g}')
        print('=' * 78)
        sigma_all = np.empty((mesh.elements.shape[0], 3), dtype=np.float64)
        vm_all = np.empty(mesh.elements.shape[0], dtype=np.float64)
        J_all = np.ones(mesh.elements.shape[0], dtype=np.float64)
        f_int = np.zeros(ndof, dtype=np.float64)
        K = sp.csr_matrix((ndof, ndof))
        for step in range(1, cfg.fem_load_steps + 1):
            factor = step / cfg.fem_load_steps
            u[fixed_dofs] = factor * fixed_values
            converged = False
            print(f'\nFEM load step {step}/{cfg.fem_load_steps}, factor={factor:.6f}')
            for it in range(1, cfg.fem_newton_max_iter + 1):
                K, f_int, sigma_all, vm_all, J_all = solve_fem('assemble_global_fem_system', mesh.nodes, mesh.elements, dofs_all, gradN_all, area_all, u.reshape(-1, 2), cfg)
                residual_free = f_int[free_dofs]
                residual_norm = float(np.linalg.norm(residual_free))
                Kff = K[free_dofs][:, free_dofs]
                du_free = spla.spsolve(Kff, -residual_free)
                du_norm = float(np.linalg.norm(du_free))
                u[free_dofs] += du_free
                u[fixed_dofs] = factor * fixed_values
                print(f'  iter {it:02d} | ||R_f||={residual_norm:.6e} | ||du_f||={du_norm:.6e} | Jmin={np.min(J_all):.6e} | Jmax={np.max(J_all):.6e}')
                if np.min(J_all) <= 0.0:
                    raise RuntimeError(f'FEM bị đảo phần tử tại step={step}, iter={it}, minJ={np.min(J_all):.6e}')
                if residual_norm < cfg.fem_newton_tol_res and du_norm < cfg.fem_newton_tol_du:
                    converged = True
                    break
            if not converged:
                raise RuntimeError(f'FEM Newton không hội tụ tại load step {step}.')
            K, f_int, sigma_all, vm_all, J_all = solve_fem('assemble_global_fem_system', mesh.nodes, mesh.elements, dofs_all, gradN_all, area_all, u.reshape(-1, 2), cfg)
            top_nodes = mesh.boundary_nodes['top']
            top_reaction_y = float(np.sum(f_int[2 * top_nodes + 1]))
            step_logs.append({'step': float(step), 'load_factor': factor, 'target_uy': factor * cfg.prescribed_uy_top, 'reaction_y': top_reaction_y, 'res_norm': residual_norm, 'du_norm': du_norm, 'J_min': float(np.min(J_all)), 'J_max': float(np.max(J_all)), 'newton_iterations': float(it)})
            print(f'  Phản lực FEM trên biên trên = {top_reaction_y:.8e}')
        wall_time = time.perf_counter() - t0
        print(f'\nHoàn tất FEM trong {wall_time:.3f} s')
        return {'u': u.reshape(-1, 2), 'K': K, 'f_int': f_int, 'sigma_elem': sigma_all, 'vm_elem': vm_all, 'J_elem': J_all, 'step_logs': step_logs, 'time': wall_time, 'area_all': area_all, 'gradN_all': gradN_all, 'dofs_all': dofs_all, 'fixed_dofs': fixed_dofs, 'fixed_values': fixed_values, 'free_dofs': free_dofs}
        return
    if operation == 'compute_fem_internal_energy':
        u = args[0] if args else kwargs['u']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        gradN_all = args[2] if len(args) > 2 else kwargs['gradN_all']
        area_all = args[3] if len(args) > 3 else kwargs['area_all']
        cfg = args[4] if len(args) > 4 else kwargs['cfg']
        total = 0.0
        for e, conn in enumerate(elements):
            grad_u = u[conn].T @ gradN_all[e]
            F = np.eye(2) + grad_u
            J = float(np.linalg.det(F))
            if J <= 0.0:
                return float('nan')
            logJ = math.log(J)
            I1 = float(np.sum(F * F))
            psi = 0.5 * cfg.mu * (I1 - 2.0) - cfg.mu * logJ + 0.5 * cfg.lam * logJ ** 2
            total += cfg.thickness * area_all[e] * psi
        return float(total)
        return
    raise ValueError(f'Unknown operation in fem: {operation}')
