# Compressible Neo-Hookean material model
from __future__ import annotations
from .common import *
from . import common

def evaluate_material(operation, *args, **kwargs):
    if operation == 'unpack_q':
        q = args[0] if args else kwargs['q']
        n_nodes = args[1] if len(args) > 1 else kwargs['n_nodes']
        return (q[:n_nodes], q[n_nodes:])
        return
    if operation == 'energy_density_and_piola':
        F11 = args[0] if args else kwargs['F11']
        F12 = args[1] if len(args) > 1 else kwargs['F12']
        F21 = args[2] if len(args) > 2 else kwargs['F21']
        F22 = args[3] if len(args) > 3 else kwargs['F22']
        cfg = args[4] if len(args) > 4 else kwargs['cfg']
        J = F11 * F22 - F12 * F21
        logJ = np.log(J)
        invT11 = F22 / J
        invT12 = -F21 / J
        invT21 = -F12 / J
        invT22 = F11 / J
        I1 = F11 * F11 + F12 * F12 + F21 * F21 + F22 * F22
        psi = 0.5 * cfg.mu * (I1 - 2.0) - cfg.mu * logJ + 0.5 * cfg.lam * logJ ** 2
        coeff = cfg.lam * logJ
        P11 = cfg.mu * (F11 - invT11) + coeff * invT11
        P12 = cfg.mu * (F12 - invT12) + coeff * invT12
        P21 = cfg.mu * (F21 - invT21) + coeff * invT21
        P22 = cfg.mu * (F22 - invT22) + coeff * invT22
        return (psi, P11, P12, P21, P22)
        return
    if operation == 'cauchy_from_piola_components':
        F11 = args[0] if args else kwargs['F11']
        F12 = args[1] if len(args) > 1 else kwargs['F12']
        F21 = args[2] if len(args) > 2 else kwargs['F21']
        F22 = args[3] if len(args) > 3 else kwargs['F22']
        P11 = args[4] if len(args) > 4 else kwargs['P11']
        P12 = args[5] if len(args) > 5 else kwargs['P12']
        P21 = args[6] if len(args) > 6 else kwargs['P21']
        P22 = args[7] if len(args) > 7 else kwargs['P22']
        J = F11 * F22 - F12 * F21
        sxx = (P11 * F11 + P12 * F12) / J
        sxy_a = (P11 * F21 + P12 * F22) / J
        sxy_b = (P21 * F11 + P22 * F12) / J
        syy = (P21 * F21 + P22 * F22) / J
        sxy = 0.5 * (sxy_a + sxy_b)
        return (sxx, sxy, syy)
        return
    if operation == 'loss_and_gradient':
        q = args[0] if args else kwargs['q']
        data = args[1] if len(args) > 1 else kwargs['data']
        ops = args[2] if len(args) > 2 else kwargs['ops']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        load_factor = args[4] if len(args) > 4 else kwargs['load_factor']
        n_nodes = data.nodes.shape[0]
        u_nodes, v_nodes = evaluate_material('unpack_q', q, n_nodes)
        ux = np.asarray(ops.DPhix_int @ u_nodes).ravel()
        uy = np.asarray(ops.DPhiy_int @ u_nodes).ravel()
        vx = np.asarray(ops.DPhix_int @ v_nodes).ravel()
        vy = np.asarray(ops.DPhiy_int @ v_nodes).ravel()
        F11 = 1.0 + ux
        F12 = uy
        F21 = vx
        F22 = 1.0 + vy
        J = F11 * F22 - F12 * F21
        minJ = float(np.min(J))
        if np.any(J <= cfg.min_J_allowed) or not np.all(np.isfinite(J)):
            return (math.inf, np.full_like(q, np.nan), {'E_in': math.nan, 'L_bottom': math.nan, 'L_top': math.nan, 'minJ': minJ, 'grad_norm': math.nan, 'mean_top_uy': math.nan})
        psi, P11, P12, P21, P22 = evaluate_material('energy_density_and_piola', F11, F12, F21, F22, cfg)
        E_in = float(np.sum(data.w_int * psi))
        u_bottom = np.asarray(ops.Phi_bottom @ u_nodes).ravel()
        v_bottom = np.asarray(ops.Phi_bottom @ v_nodes).ravel()
        wb_sum = float(np.sum(data.w_bottom))
        L_bottom = float(np.sum(data.w_bottom * (u_bottom ** 2 + v_bottom ** 2)) / wb_sum)
        target_top_uy = load_factor * cfg.prescribed_uy_top
        v_top = np.asarray(ops.Phi_top @ v_nodes).ravel()
        top_error = v_top - target_top_uy
        wt_sum = float(np.sum(data.w_top))
        L_top = float(np.sum(data.w_top * top_error ** 2) / wt_sum)
        loss = E_in + cfg.alpha_bottom * L_bottom + cfg.alpha_top * L_top
        grad_u = np.asarray(ops.DPhix_int.T @ (data.w_int * P11)).ravel()
        grad_u += np.asarray(ops.DPhiy_int.T @ (data.w_int * P12)).ravel()
        grad_v = np.asarray(ops.DPhix_int.T @ (data.w_int * P21)).ravel()
        grad_v += np.asarray(ops.DPhiy_int.T @ (data.w_int * P22)).ravel()
        grad_u += cfg.alpha_bottom * (2.0 / wb_sum) * np.asarray(ops.Phi_bottom.T @ (data.w_bottom * u_bottom)).ravel()
        grad_v += cfg.alpha_bottom * (2.0 / wb_sum) * np.asarray(ops.Phi_bottom.T @ (data.w_bottom * v_bottom)).ravel()
        grad_v += cfg.alpha_top * (2.0 / wt_sum) * np.asarray(ops.Phi_top.T @ (data.w_top * top_error)).ravel()
        grad = np.concatenate([grad_u, grad_v])
        return (loss, grad, {'E_in': E_in, 'L_bottom': L_bottom, 'L_top': L_top, 'minJ': minJ, 'grad_norm': float(np.linalg.norm(grad)), 'mean_top_uy': float(np.sum(data.w_top * v_top) / wt_sum)})
        return
    if operation == 'evaluate_fields':
        q = args[0] if args else kwargs['q']
        x = args[1] if len(args) > 1 else kwargs['x']
        Phi = args[2] if len(args) > 2 else kwargs['Phi']
        DPhix = args[3] if len(args) > 3 else kwargs['DPhix']
        DPhiy = args[4] if len(args) > 4 else kwargs['DPhiy']
        cfg = args[5] if len(args) > 5 else kwargs['cfg']
        n_nodes = Phi.shape[1]
        u_nodes, v_nodes = evaluate_material('unpack_q', q, n_nodes)
        u = np.asarray(Phi @ u_nodes).ravel()
        v = np.asarray(Phi @ v_nodes).ravel()
        ux = np.asarray(DPhix @ u_nodes).ravel()
        uy = np.asarray(DPhiy @ u_nodes).ravel()
        vx = np.asarray(DPhix @ v_nodes).ravel()
        vy = np.asarray(DPhiy @ v_nodes).ravel()
        F11 = 1.0 + ux
        F12 = uy
        F21 = vx
        F22 = 1.0 + vy
        J = F11 * F22 - F12 * F21
        _, P11, P12, P21, P22 = evaluate_material('energy_density_and_piola', F11, F12, F21, F22, cfg)
        sxx, sxy, syy = evaluate_material('cauchy_from_piola_components', F11, F12, F21, F22, P11, P12, P21, P22)
        vm = np.sqrt(np.maximum(0.0, sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2))
        umag = np.sqrt(u ** 2 + v ** 2)
        x_def = x + cfg.deformation_scale * np.column_stack([u, v])
        return Fields(u, v, ux, uy, vx, vy, J, umag, sxx, sxy, syy, vm, x_def)
        return
    if operation == 'evaluate_top_reaction':
        q = args[0] if args else kwargs['q']
        data = args[1] if len(args) > 1 else kwargs['data']
        ops = args[2] if len(args) > 2 else kwargs['ops']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        n_nodes = data.nodes.shape[0]
        u_nodes, v_nodes = evaluate_material('unpack_q', q, n_nodes)
        ux = np.asarray(ops.DPhix_top @ u_nodes).ravel()
        uy = np.asarray(ops.DPhiy_top @ u_nodes).ravel()
        vx = np.asarray(ops.DPhix_top @ v_nodes).ravel()
        vy = np.asarray(ops.DPhiy_top @ v_nodes).ravel()
        F11 = 1.0 + ux
        F12 = uy
        F21 = vx
        F22 = 1.0 + vy
        J = F11 * F22 - F12 * F21
        if np.any(J <= cfg.min_J_allowed):
            return float('nan')
        _, _, _, _, P22 = evaluate_material('energy_density_and_piola', F11, F12, F21, F22, cfg)
        return float(np.sum(data.w_top * P22))
        return
    raise ValueError(f'Unknown operation in material: {operation}')
