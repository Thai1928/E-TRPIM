# TRPIM optimization and deformation safeguards
from __future__ import annotations
from .common import *
from . import common
from . import material

def optimize_trpim(operation, *args, **kwargs):
    if operation == 'impose_optional_hard_boundary':
        q = args[0] if args else kwargs['q']
        data = args[1] if len(args) > 1 else kwargs['data']
        cfg = args[2] if len(args) > 2 else kwargs['cfg']
        load_factor = args[3] if len(args) > 3 else kwargs['load_factor']
        if not cfg.hard_impose_boundary_nodes:
            return
        n_nodes = data.nodes.shape[0]
        q[data.fixed_bottom_nodes] = 0.0
        q[n_nodes + data.fixed_bottom_nodes] = 0.0
        q[n_nodes + data.top_nodes] = load_factor * cfg.prescribed_uy_top
        return
    if operation == 'hard_boundary_dofs':
        data = args[0] if args else kwargs['data']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        if not cfg.hard_impose_boundary_nodes:
            return np.empty(0, dtype=np.int64)
        n_nodes = data.nodes.shape[0]
        return np.unique(np.concatenate([data.fixed_bottom_nodes, n_nodes + data.fixed_bottom_nodes, n_nodes + data.top_nodes])).astype(np.int64)
        return
    if operation == 'min_j_from_derivative_operators':
        q = args[0] if args else kwargs['q']
        DPhix = args[1] if len(args) > 1 else kwargs['DPhix']
        DPhiy = args[2] if len(args) > 2 else kwargs['DPhiy']
        n_nodes = DPhix.shape[1]
        u_nodes, v_nodes = material.evaluate_material('unpack_q', q, n_nodes)
        ux = np.asarray(DPhix @ u_nodes).ravel()
        uy = np.asarray(DPhiy @ u_nodes).ravel()
        vx = np.asarray(DPhix @ v_nodes).ravel()
        vy = np.asarray(DPhiy @ v_nodes).ravel()
        J = (1.0 + ux) * (1.0 + vy) - uy * vx
        if not np.all(np.isfinite(J)):
            return -math.inf
        return float(np.min(J))
        return
    if operation == 'min_deformed_triangle_area_ratio':
        q = args[0] if args else kwargs['q']
        data = args[1] if len(args) > 1 else kwargs['data']
        n_nodes = data.nodes.shape[0]
        u_nodes, v_nodes = material.evaluate_material('unpack_q', q, n_nodes)
        deformed = data.nodes + np.column_stack([u_nodes, v_nodes])
        X0 = data.nodes[data.elements]
        Xd = deformed[data.elements]
        cross0 = (X0[:, 1, 0] - X0[:, 0, 0]) * (X0[:, 2, 1] - X0[:, 0, 1]) - (X0[:, 2, 0] - X0[:, 0, 0]) * (X0[:, 1, 1] - X0[:, 0, 1])
        crossd = (Xd[:, 1, 0] - Xd[:, 0, 0]) * (Xd[:, 2, 1] - Xd[:, 0, 1]) - (Xd[:, 2, 0] - Xd[:, 0, 0]) * (Xd[:, 1, 1] - Xd[:, 0, 1])
        if np.any(np.abs(cross0) <= 1e-16):
            raise ValueError('Lưới có tam giác suy biến khi kiểm tra geometry guard.')
        ratio = crossd / cross0
        if not np.all(np.isfinite(ratio)):
            return -math.inf
        return float(np.min(ratio))
        return
    if operation == 'make_continuation_initial_guess':
        q_previous = args[0] if args else kwargs['q_previous']
        previous_target = args[1] if len(args) > 1 else kwargs['previous_target']
        target = args[2] if len(args) > 2 else kwargs['target']
        load_factor = args[3] if len(args) > 3 else kwargs['load_factor']
        data = args[4] if len(args) > 4 else kwargs['data']
        ops = args[5] if len(args) > 5 else kwargs['ops']
        cfg = args[6] if len(args) > 6 else kwargs['cfg']
        n_nodes = data.nodes.shape[0]
        eta = np.clip(data.nodes[:, 1] / cfg.Ly, 0.0, 1.0)
        q_affine = np.zeros_like(q_previous)
        q_affine[n_nodes:] = target * eta
        if previous_target <= 0.0:
            q_guess = q_affine
        else:
            q_scaled = q_previous * (target / previous_target)
            blend = float(np.clip(cfg.continuation_blend, 0.0, 1.0))
            q_guess = q_affine.copy()
            while blend >= 0.0001:
                trial = blend * q_scaled + (1.0 - blend) * q_affine
                optimize_trpim('impose_optional_hard_boundary', trial, data, cfg, load_factor)
                area_ratio = optimize_trpim('min_deformed_triangle_area_ratio', trial, data)
                min_j_guard = optimize_trpim('min_j_from_derivative_operators', trial, ops.DPhix_guard, ops.DPhiy_guard)
                if area_ratio >= cfg.min_deformed_area_ratio and min_j_guard >= cfg.min_guard_J:
                    q_guess = trial
                    break
                blend *= 0.5
        optimize_trpim('impose_optional_hard_boundary', q_guess, data, cfg, load_factor)
        return q_guess
        return
    if operation == 'solve_etrpim':
        data = args[0] if args else kwargs['data']
        ops = args[1] if len(args) > 1 else kwargs['ops']
        cfg = args[2] if len(args) > 2 else kwargs['cfg']
        n_nodes = data.nodes.shape[0]
        q = np.zeros(2 * n_nodes, dtype=np.float64)
        m1 = np.zeros_like(q)
        m2 = np.zeros_like(q)
        fixed_dofs = optimize_trpim('hard_boundary_dofs', data, cfg)
        history = np.zeros((cfg.total_iterations, 15), dtype=np.float64)
        step_history: List[List[float]] = []
        global_iter = 0
        adam_iter = 0
        previous_target = 0.0
        t0 = time.perf_counter()
        print('\n' + '=' * 78)
        print('ETRPIM ổn định / point-cloud cho tấm có 3 lỗ')
        print(f'Số nút trường       : {n_nodes}')
        print(f'Số điểm mẫu miền    : {data.x_int.shape[0]}')
        print(f'Samples / node      : {data.x_int.shape[0] / n_nodes:.3f}')
        print(f'Số guard points     : {data.x_guard.shape[0]}')
        print(f'Số ẩn tối ưu        : {q.size}')
        print(f'Tổng vòng lặp Adam  : {cfg.total_iterations}')
        print(f'Visibility criterion: {cfg.use_visibility_criterion}')
        print(f'Hard boundary nodes : {cfg.hard_impose_boundary_nodes}')
        print('Không dùng Gauss cell, không lắp ma trận tiếp tuyến TRPIM.')
        print('=' * 78)
        for step in range(1, cfg.load_steps + 1):
            load_factor = step / cfg.load_steps
            target_uy = load_factor * cfg.prescribed_uy_top
            local_iterations = 0
            q = optimize_trpim('make_continuation_initial_guess', q, previous_target, target_uy, load_factor, data, ops, cfg)
            previous_target = target_uy
            print(f'\nBước chuyển vị {step}/{cfg.load_steps}: u_y(top) mục tiêu = {target_uy:.8g}')
            if cfg.reset_adam_each_load_step:
                m1.fill(0.0)
                m2.fill(0.0)
                adam_iter = 0
            effective_lr = cfg.lr
            rejected_updates = 0
            last_parts: Dict[str, float] = {}
            last_loss = math.nan
            last_guard_j = math.nan
            last_area_ratio = math.nan
            for it in range(1, cfg.iter_per_load_step + 1):
                global_iter += 1
                adam_iter += 1
                local_iterations = it
                loss, grad, parts = material.evaluate_material('loss_and_gradient', q, data, ops, cfg, load_factor)
                if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
                    raise FloatingPointError('Trạng thái hiện tại đã có loss/gradient không hữu hạn. Hãy kiểm tra support diagnostics hoặc giảm bước tải.')
                if fixed_dofs.size:
                    grad[fixed_dofs] = 0.0
                    m1[fixed_dofs] = 0.0
                    m2[fixed_dofs] = 0.0
                raw_grad_norm = float(np.linalg.norm(grad))
                if raw_grad_norm > cfg.grad_clip_norm:
                    grad *= cfg.grad_clip_norm / raw_grad_norm
                m1 = cfg.beta1 * m1 + (1.0 - cfg.beta1) * grad
                m2 = cfg.beta2 * m2 + (1.0 - cfg.beta2) * (grad * grad)
                mhat = m1 / (1.0 - cfg.beta1 ** adam_iter)
                vhat = m2 / (1.0 - cfg.beta2 ** adam_iter)
                dq = effective_lr * mhat / (np.sqrt(vhat) + cfg.adam_eps)
                if fixed_dofs.size:
                    dq[fixed_dofs] = 0.0
                max_dq = float(np.max(np.abs(dq)))
                if max_dq > cfg.max_coeff_update:
                    dq *= cfg.max_coeff_update / max_dq
                accepted = False
                accepted_scale = 0.0
                accepted_loss = loss
                accepted_parts = parts
                accepted_guard_j = optimize_trpim('min_j_from_derivative_operators', q, ops.DPhix_guard, ops.DPhiy_guard)
                accepted_area_ratio = optimize_trpim('min_deformed_triangle_area_ratio', q, data)
                for bt in range(cfg.max_backtracking + 1):
                    scale = cfg.backtracking_factor ** bt
                    q_trial = q - scale * dq
                    optimize_trpim('impose_optional_hard_boundary', q_trial, data, cfg, load_factor)
                    area_ratio = optimize_trpim('min_deformed_triangle_area_ratio', q_trial, data)
                    if area_ratio < cfg.min_deformed_area_ratio:
                        continue
                    min_j_guard = optimize_trpim('min_j_from_derivative_operators', q_trial, ops.DPhix_guard, ops.DPhiy_guard)
                    if min_j_guard < cfg.min_guard_J:
                        continue
                    trial_loss, _, trial_parts = material.evaluate_material('loss_and_gradient', q_trial, data, ops, cfg, load_factor)
                    tolerance = cfg.allow_relative_loss_increase * max(1.0, abs(loss))
                    if np.isfinite(trial_loss) and trial_loss <= loss + tolerance:
                        q = q_trial
                        accepted = True
                        accepted_scale = scale
                        accepted_loss = trial_loss
                        accepted_parts = trial_parts
                        accepted_guard_j = min_j_guard
                        accepted_area_ratio = area_ratio
                        break
                if accepted:
                    effective_lr = min(cfg.lr, effective_lr * 1.002)
                else:
                    rejected_updates += 1
                    effective_lr = max(cfg.min_lr, 0.5 * effective_lr)
                    m1.fill(0.0)
                    m2.fill(0.0)
                    adam_iter = 0
                history[global_iter - 1] = [global_iter, step, load_factor, target_uy, accepted_loss, accepted_parts['E_in'], accepted_parts['L_bottom'], accepted_parts['L_top'], accepted_parts['minJ'], accepted_parts['mean_top_uy'], raw_grad_norm, accepted_guard_j, accepted_area_ratio, accepted_scale, effective_lr]
                last_parts = accepted_parts
                last_loss = accepted_loss
                last_guard_j = accepted_guard_j
                last_area_ratio = accepted_area_ratio
                if it == 1 or it % cfg.print_every == 0:
                    status = 'OK' if accepted else 'REJECT'
                    print(f"  iter {global_iter:7d}/{cfg.total_iterations:7d} | loss={accepted_loss:.6e} | E={accepted_parts['E_in']:.6e} | minJ(sample)={accepted_parts['minJ']:.5g} | minJ(guard)={accepted_guard_j:.5g} | area_ratio={accepted_area_ratio:.5g} | lr={effective_lr:.3e} | {status}")
                if cfg.grad_tol > 0.0 and raw_grad_norm < cfg.grad_tol:
                    print(f'  Dừng sớm tại bước {step}, local iter={it}, |g|={raw_grad_norm:.3e}')
                    break
            reaction = material.evaluate_material('evaluate_top_reaction', q, data, ops, cfg)
            step_history.append([step, load_factor, target_uy, last_parts.get('mean_top_uy', math.nan), reaction, last_parts.get('minJ', math.nan), last_loss, local_iterations, last_guard_j, last_area_ratio, rejected_updates])
            print(f'  Phản lực ETRPIM = {reaction:.8e} | minJ_guard={last_guard_j:.6g} | min_area_ratio={last_area_ratio:.6g} | rejected={rejected_updates}')
        wall_time = time.perf_counter() - t0
        print(f'\nHoàn tất ETRPIM trong {wall_time:.3f} s')
        return {'q': q, 'history': history[:global_iter], 'step_history': np.asarray(step_history, dtype=np.float64), 'time': wall_time, 'method': 'Stable conventional TRPIM + visibility + point-cloud'}
        return
    raise ValueError(f'Unknown operation in trpim: {operation}')
