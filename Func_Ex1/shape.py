# TRPIM interpolation and shape matrices
from __future__ import annotations
from .common import *
from . import common

def evaluate_shapes(operation, *args, **kwargs):
    if operation == 'quartic_rbf_and_derivative':
        dx = args[0] if args else kwargs['dx']
        dy = args[1] if len(args) > 1 else kwargs['dy']
        radius = args[2] if len(args) > 2 else kwargs['radius']
        r = np.sqrt(dx * dx + dy * dy)
        radius = max(radius, 1e-14)
        q = r / radius
        val = np.zeros_like(q)
        dxv = np.zeros_like(q)
        dyv = np.zeros_like(q)
        inside = q <= 1.0
        if not np.any(inside):
            return (val, dxv, dyv)
        qi = q[inside]
        val[inside] = 1.0 - 6.0 * qi ** 2 + 8.0 * qi ** 3 - 3.0 * qi ** 4
        dR_dq = -12.0 * qi + 24.0 * qi ** 2 - 12.0 * qi ** 3
        dR_dr = dR_dq / radius
        ids = np.where(inside)[0]
        ri = r[inside]
        nonzero = ri > 1e-14
        if np.any(nonzero):
            ii = ids[nonzero]
            dxv[ii] = dR_dr[nonzero] * dx[ii] / ri[nonzero]
            dyv[ii] = dR_dr[nonzero] * dy[ii] / ri[nonzero]
        return (val, dxv, dyv)
        return
    if operation == 'chebyshev_basis_2d':
        x = args[0] if args else kwargs['x']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        xb = 2.0 * x[0] / cfg.Lx - 1.0
        yb = 2.0 * x[1] / cfg.Ly - 1.0
        dxb = 2.0 / cfg.Lx
        dyb = 2.0 / cfg.Ly
        Tx = [1.0, xb, 2.0 * xb * xb - 1.0]
        Ty = [1.0, yb, 2.0 * yb * yb - 1.0]
        dTx = [0.0, 1.0, 4.0 * xb]
        dTy = [0.0, 1.0, 4.0 * yb]
        terms = [(0, 0), (1, 0), (0, 1)]
        if cfg.cheb_order >= 2:
            terms += [(2, 0), (1, 1), (0, 2)]
        if cfg.cheb_order > 2:
            raise ValueError('cheb_order chỉ hỗ trợ 1 hoặc 2.')
        p, px, py = ([], [], [])
        for i, j in terms:
            p.append(Tx[i] * Ty[j])
            px.append(dTx[i] * dxb * Ty[j])
            py.append(Tx[i] * dTy[j] * dyb)
        return (np.asarray(p), np.asarray(px), np.asarray(py))
        return
    if operation == 'chebyshev_matrix_at_nodes':
        nodes = args[0] if args else kwargs['nodes']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        return np.vstack([evaluate_shapes('chebyshev_basis_2d', x, cfg)[0] for x in nodes])
        return
    if operation == 'visible_support_candidates':
        x = args[0] if args else kwargs['x']
        nodes = args[1] if len(args) > 1 else kwargs['nodes']
        tree = args[2] if len(args) > 2 else kwargs['tree']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        required = args[4] if len(args) > 4 else kwargs['required']
        n_total = nodes.shape[0]
        pool = min(n_total, max(required, cfg.support_pool_factor * cfg.max_support_nodes))
        while True:
            _, idx = tree.query(x, k=pool)
            idx = np.atleast_1d(idx).astype(np.int64)
            if not cfg.use_visibility_criterion:
                return idx
            pts = nodes[idx]
            d = pts - x[None, :]
            d2 = np.sum(d * d, axis=1)
            visible = np.ones(idx.size, dtype=bool)
            for hole in cfg.holes:
                c = np.array([hole['cx'], hole['cy']], dtype=np.float64)
                radius = float(hole['R'])
                numer = np.sum((c[None, :] - x[None, :]) * d, axis=1)
                t = np.divide(numer, d2, out=np.zeros_like(numer), where=d2 > 1e-30)
                t = np.clip(t, 0.0, 1.0)
                closest = x[None, :] + t[:, None] * d
                dist2 = np.sum((closest - c[None, :]) ** 2, axis=1)
                visible &= dist2 >= (radius - 1e-10) ** 2
            idx_visible = idx[visible]
            if idx_visible.size >= required:
                return idx_visible
            if pool >= n_total:
                raise RuntimeError(f'Không tìm đủ {required} nút support nhìn thấy tại x={x}; chỉ có {idx_visible.size}. Hãy giảm support_nodes hoặc làm mịn lưới.')
            pool = min(n_total, max(pool + 1, 2 * pool))
        return
    if operation == 'trpim_shape_at_point':
        x = args[0] if args else kwargs['x']
        nodes = args[1] if len(args) > 1 else kwargs['nodes']
        tree = args[2] if len(args) > 2 else kwargs['tree']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        n_total = nodes.shape[0]
        n_poly = 3 if cfg.cheb_order == 1 else 6
        min_required = n_poly + 3
        k_start = min(max(cfg.support_nodes, min_required), n_total)
        k_end = min(max(cfg.max_support_nodes, k_start), n_total)
        visible_idx = evaluate_shapes('visible_support_candidates', x, nodes, tree, cfg, k_start)
        k_end = min(k_end, int(visible_idx.size))
        candidates = list(range(k_start, k_end + 1, 2))
        if not candidates or candidates[-1] != k_end:
            candidates.append(k_end)
        last_error: Exception | None = None
        best_data = None
        best_cond = math.inf
        for k in candidates:
            idx = visible_idx[:k]
            xs = nodes[idx]
            n = xs.shape[0]
            dxq = xs[:, None, 0] - xs[None, :, 0]
            dyq = xs[:, None, 1] - xs[None, :, 1]
            pairwise_max = float(np.max(np.sqrt(dxq * dxq + dyq * dyq)))
            max_eval_dist = float(np.max(np.linalg.norm(xs - x, axis=1)))
            radius = max(cfg.base_support_radius, 1.001 * max_eval_dist, 1e-08)
            if cfg.rbf_cover_pairwise_support:
                radius = max(radius, 1.001 * pairwise_max)
            Rq, _, _ = evaluate_shapes('quartic_rbf_and_derivative', dxq.ravel(), dyq.ravel(), radius)
            Rq = Rq.reshape(n, n)
            Tq = evaluate_shapes('chebyshev_matrix_at_nodes', xs, cfg)
            G = np.zeros((n + n_poly, n + n_poly), dtype=np.float64)
            G[:n, :n] = Rq
            G[:n, n:] = Tq
            G[n:, :n] = Tq.T
            if cfg.moment_regularization > 0.0:
                G[:n, :n] += cfg.moment_regularization * np.eye(n)
            try:
                cond = float(np.linalg.cond(G))
                if not np.isfinite(cond):
                    raise np.linalg.LinAlgError('condition number không hữu hạn')
                rhs = np.zeros((n + n_poly, n), dtype=np.float64)
                rhs[:n, :] = np.eye(n)
                coeff = np.linalg.solve(G, rhs)
                dx = x[0] - xs[:, 0]
                dy = x[1] - xs[:, 1]
                r, rx, ry = evaluate_shapes('quartic_rbf_and_derivative', dx, dy, radius)
                t, tx, ty = evaluate_shapes('chebyshev_basis_2d', x, cfg)
                phi = np.concatenate([r, t]) @ coeff
                phix = np.concatenate([rx, tx]) @ coeff
                phiy = np.concatenate([ry, ty]) @ coeff
                max_phi = float(np.max(np.abs(phi)))
                grad_scaled = cfg.node_spacing_estimate * float(max(np.max(np.abs(phix)), np.max(np.abs(phiy))))
                valid_shape = max_phi <= cfg.shape_value_limit and grad_scaled <= cfg.shape_gradient_limit_scaled
                if cond < best_cond and valid_shape:
                    best_cond = cond
                    best_data = (idx, phi, phix, phiy)
                if cond <= cfg.cond_limit and valid_shape:
                    return (idx, phi, phix, phiy)
                last_error = np.linalg.LinAlgError(f'support chưa đạt: cond={cond:.3e}, max|phi|={max_phi:.3e}, h|max grad|={grad_scaled:.3e}')
            except Exception as exc:
                last_error = exc
        if best_data is not None:
            return best_data
        raise np.linalg.LinAlgError(f'Không xây dựng được TRPIM support ổn định tại x={x}. Lỗi cuối: {last_error}')
        return
    if operation == 'build_shape_matrices':
        x_eval = args[0] if args else kwargs['x_eval']
        nodes = args[1] if len(args) > 1 else kwargs['nodes']
        cfg = args[2] if len(args) > 2 else kwargs['cfg']
        tree = cKDTree(nodes)
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        vals_x: List[np.ndarray] = []
        vals_y: List[np.ndarray] = []
        for i, x in enumerate(x_eval):
            idx, phi, phix, phiy = evaluate_shapes('trpim_shape_at_point', x, nodes, tree, cfg)
            keep = (np.abs(phi) > 1e-14) | (np.abs(phix) > 1e-14) | (np.abs(phiy) > 1e-14)
            idx = idx[keep]
            phi = phi[keep]
            phix = phix[keep]
            phiy = phiy[keep]
            rows.append(np.full(idx.size, i, dtype=np.int64))
            cols.append(idx)
            vals.append(phi)
            vals_x.append(phix)
            vals_y.append(phiy)
        I = np.concatenate(rows)
        J = np.concatenate(cols)
        shape = (x_eval.shape[0], nodes.shape[0])
        Phi = sparse.coo_matrix((np.concatenate(vals), (I, J)), shape=shape).tocsr()
        DPhix = sparse.coo_matrix((np.concatenate(vals_x), (I, J)), shape=shape).tocsr()
        DPhiy = sparse.coo_matrix((np.concatenate(vals_y), (I, J)), shape=shape).tocsr()
        return (Phi, DPhix, DPhiy)
        return
    if operation == 'build_operators':
        data = args[0] if args else kwargs['data']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        print('\nĐang xây dựng ma trận TRPIM bằng neighbour sample point searching ...')
        t0 = time.perf_counter()
        Phi_int, DPhix_int, DPhiy_int = evaluate_shapes('build_shape_matrices', data.x_int, data.nodes, cfg)
        Phi_bottom, _, _ = evaluate_shapes('build_shape_matrices', data.x_bottom, data.nodes, cfg)
        Phi_top, DPhix_top, DPhiy_top = evaluate_shapes('build_shape_matrices', data.x_top, data.nodes, cfg)
        _, DPhix_guard, DPhiy_guard = evaluate_shapes('build_shape_matrices', data.x_guard, data.nodes, cfg)
        print(f'Hoàn tất trong {time.perf_counter() - t0:.2f} s | Phi_int={Phi_int.shape}, nnz={Phi_int.nnz}, guard_points={data.x_guard.shape[0]}')
        return Operators(Phi_int=Phi_int, DPhix_int=DPhix_int, DPhiy_int=DPhiy_int, Phi_bottom=Phi_bottom, Phi_top=Phi_top, DPhix_top=DPhix_top, DPhiy_top=DPhiy_top, DPhix_guard=DPhix_guard, DPhiy_guard=DPhiy_guard)
        return
    raise ValueError(f'Unknown operation in shape: {operation}')
