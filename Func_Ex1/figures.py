# Figures and profile comparisons
from __future__ import annotations
from .common import *
from . import common
from . import material
from . import shape

def create_figures(operation, *args, **kwargs):
    if operation == 'colorbar_limits_ticks':
        val = args[0] if args else kwargs['val']
        n_ticks = args[1] if len(args) > 1 else kwargs.get('n_ticks', 6)
        val = np.asarray(val, dtype=float)
        finite = val[np.isfinite(val)]
        if finite.size == 0:
            raise ValueError('Không có giá trị hữu hạn để tạo colorbar.')
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if np.isclose(vmin, vmax):
            delta = max(abs(vmin) * 1e-06, 1e-12)
            vmin -= delta
            vmax += delta
        ticks = np.linspace(vmin, vmax, n_ticks)
        return (np.ma.masked_invalid(val), vmin, vmax, ticks)
        return
    if operation == 'format_colorbar_tick':
        x = args[0] if args else kwargs['x']
        if x == 0.0:
            return '0.0000'
        if 0.001 <= abs(x) < 10000.0:
            return f'{x:.4f}'
        return f'{x:.3e}'
        return
    if operation == 'add_minmax_colorbar':
        fig = args[0] if args else kwargs['fig']
        ax = args[1] if len(args) > 1 else kwargs['ax']
        mappable = args[2] if len(args) > 2 else kwargs['mappable']
        label = args[3] if len(args) > 3 else kwargs['label']
        ticks = args[4] if len(args) > 4 else kwargs['ticks']
        shrink = args[5] if len(args) > 5 else kwargs.get('shrink', None)
        kwargs = {'ticks': ticks}
        if shrink is not None:
            kwargs['shrink'] = shrink
        cb = fig.colorbar(mappable, ax=ax, **kwargs)
        cb.ax.set_yticklabels([create_figures('format_colorbar_tick', float(x)) for x in ticks])
        cb.set_label(label)
        return cb
        return
    if operation == 'set_equal_axis':
        ax = args[0] if args else kwargs['ax']
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return
    if operation == 'draw_reference_outline':
        ax = args[0] if args else kwargs['ax']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        linestyle = args[2] if len(args) > 2 else kwargs.get('linestyle', 'k--')
        outer = np.array([[0.0, 0.0], [cfg.Lx, 0.0], [cfg.Lx, cfg.Ly], [0.0, cfg.Ly], [0.0, 0.0]])
        ax.plot(outer[:, 0], outer[:, 1], linestyle, lw=0.9)
        theta = np.linspace(0.0, 2.0 * math.pi, 240)
        for hole in cfg.holes:
            x = hole['cx'] + hole['R'] * np.cos(theta)
            y = hole['cy'] + hole['R'] * np.sin(theta)
            ax.plot(x, y, linestyle, lw=0.9)
        return
    if operation == 'draw_deformed_boundary_edges':
        ax = args[0] if args else kwargs['ax']
        deformed_nodes = args[1] if len(args) > 1 else kwargs['deformed_nodes']
        boundary_edges = args[2] if len(args) > 2 else kwargs['boundary_edges']
        color = args[3] if len(args) > 3 else kwargs.get('color', 'black')
        linewidth = args[4] if len(args) > 4 else kwargs.get('linewidth', 0.9)
        for edges in boundary_edges.values():
            for edge in edges:
                p = deformed_nodes[edge]
                ax.plot(p[:, 0], p[:, 1], color=color, lw=linewidth)
        return
    if operation == 'save_fig01_trpim_nodes_samples':
        data = args[0] if args else kwargs['data']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        out = args[2] if len(args) > 2 else kwargs['out']
        if not cfg.save_figures:
            return
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        create_figures('draw_reference_outline', ax, cfg, linestyle='k-')
        ax.scatter(data.nodes[:, 0], data.nodes[:, 1], s=20, marker='o', color='black', label='TRPIM field nodes')
        ax.scatter(data.x_int[:, 0], data.x_int[:, 1], s=2, marker='s', color='red', label='Integration nodes')
        ax.scatter(data.x_bottom[:, 0], data.x_bottom[:, 1], s=2, marker='s', color='red')
        ax.scatter(data.x_top[:, 0], data.x_top[:, 1], s=2, marker='s', color='red')
        create_figures('set_equal_axis', ax)
        ax.set_title('')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(out / 'Fig01_1_ETRPIM_nodes_samples_BC.png', dpi=1000)
        plt.close(fig)
        return
    if operation == 'save_fig01_fem_mesh':
        mesh = args[0] if args else kwargs['mesh']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        out = args[2] if len(args) > 2 else kwargs['out']
        if not cfg.save_figures:
            return
        fig, ax = plt.subplots(figsize=cfg.field_figsize)
        for conn in mesh.elements:
            xy = mesh.nodes[conn]
            xy = np.vstack([xy, xy[0]])
            ax.plot(xy[:, 0], xy[:, 1], color='black', lw=0.25)
        create_figures('set_equal_axis', ax)
        ax.set_title('FEM meshing')
        fig.tight_layout()
        fig.savefig(out / 'Fig01_2_FEM_meshing.png', dpi=1000)
        plt.close(fig)
        return
    if operation == 'save_fig01_boundary_classification':
        mesh = args[0] if args else kwargs['mesh']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        out = args[2] if len(args) > 2 else kwargs['out']
        if not cfg.save_figures:
            return
        colors = {'left': '#C44E52', 'right': '#4C72B0', 'top': '#55A868', 'bottom': '#DD8452', 'hole': '#8172B2'}
        fig, ax = plt.subplots(figsize=cfg.field_figsize)
        for name in ['left', 'right', 'top', 'bottom', 'hole']:
            first = True
            for edge in mesh.boundary_edges[name]:
                p = mesh.nodes[edge]
                ax.plot(p[:, 0], p[:, 1], color=colors[name], lw=2.0, label=name.capitalize() if first else None)
                first = False
        create_figures('set_equal_axis', ax)
        ax.set_title('Boundary classification')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)
        fig.tight_layout()
        fig.savefig(out / 'Fig01_3_boundary_classification.png', dpi=300)
        plt.close(fig)
        return
    if operation == 'save_load_displacement_curves':
        trpim_step_history = args[0] if args else kwargs['trpim_step_history']
        fem_step_logs = args[1] if len(args) > 1 else kwargs['fem_step_logs']
        out = args[2] if len(args) > 2 else kwargs['out']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        if not cfg.save_figures:
            return
        fem_uy = np.array([r['target_uy'] for r in fem_step_logs])
        fem_reaction = np.array([r['reaction_y'] for r in fem_step_logs])
        tr_uy = trpim_step_history[:, 2]
        tr_reaction = trpim_step_history[:, 4]
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(tr_uy, tr_reaction, 'o-', lw=1.7, label='ETRPIM')
        ax.set_xlabel('Prescribed top displacement')
        ax.set_ylabel('Vertical reaction')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig02_1_ETRPIM_load_displacement.png', dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(fem_uy, fem_reaction, 's-', lw=1.7, label='FEM')
        ax.set_xlabel('Prescribed top displacement')
        ax.set_ylabel('Vertical reaction')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig02_2_FEM_load_displacement.png', dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(tr_uy, tr_reaction, 'o-', lw=1.7, label='ETRPIM')
        ax.plot(fem_uy, fem_reaction, 's--', lw=1.5, label='FEM')
        ax.set_xlabel('Prescribed top displacement')
        ax.set_ylabel('Vertical reaction')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig02_3_ETRPIM_vs_FEM.png', dpi=300)
        plt.close(fig)
        return
    if operation == 'save_training_history':
        history = args[0] if args else kwargs['history']
        fem_step_logs = args[1] if len(args) > 1 else kwargs['fem_step_logs']
        cfg = args[2] if len(args) > 2 else kwargs['cfg']
        out = args[3] if len(args) > 3 else kwargs['out']
        if not cfg.save_figures:
            return
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(history[:, 0], history[:, 4], label='loss')
        ax.plot(history[:, 0], history[:, 5], '--', label='E_in')
        ax.plot(history[:, 0], cfg.alpha_bottom * history[:, 6], ':', label='alpha_bottom L_bottom')
        ax.plot(history[:, 0], cfg.alpha_top * history[:, 7], ':', label='alpha_top L_top')
        ax.axhline(0.0, color='black', lw=0.6)
        ax.grid(True)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Signed value')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig03_1_ETRPIM_training_history_signed.png', dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(history[:, 0], history[:, 8], label='min J at energy samples')
        if history.shape[1] > 11:
            ax.plot(history[:, 0], history[:, 11], '--', label='min J at guard points')
        ax.axhline(cfg.min_guard_J, color='black', lw=0.8, ls=':', label='guard limit')
        ax.grid(True)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('min J')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig04_1_ETRPIM_minJ_history.png', dpi=300)
        plt.close(fig)
        if history.shape[1] > 12:
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            ax.plot(history[:, 0], history[:, 12], label='minimum deformed area ratio')
            ax.axhline(cfg.min_deformed_area_ratio, color='black', lw=0.8, ls=':', label='geometry guard limit')
            ax.grid(True)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('$A_{def}/A_0$')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / 'Fig04_3_ETRPIM_geometry_guard.png', dpi=300)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        step = np.array([r['step'] for r in fem_step_logs])
        jmin = np.array([r['J_min'] for r in fem_step_logs])
        jmax = np.array([r['J_max'] for r in fem_step_logs])
        ax.plot(step, jmin, 'o-', label='FEM min J')
        ax.plot(step, jmax, 's--', label='FEM max J')
        ax.grid(True)
        ax.set_xlabel('Load step')
        ax.set_ylabel('J')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig04_2_FEM_J_history.png', dpi=300)
        plt.close(fig)
        return
    if operation == 'plot_field_on_deformed':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        u_field = args[2] if len(args) > 2 else kwargs['u_field']
        field_value = args[3] if len(args) > 3 else kwargs['field_value']
        boundary_edges = args[4] if len(args) > 4 else kwargs['boundary_edges']
        cfg = args[5] if len(args) > 5 else kwargs['cfg']
        save_path = args[6] if len(args) > 6 else kwargs['save_path']
        label = args[7] if len(args) > 7 else kwargs['label']
        title = args[8] if len(args) > 8 else kwargs.get('title', None)
        if not cfg.save_figures:
            return
        deformed = nodes + cfg.deformation_scale * u_field
        triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], elements)
        value_plot, vmin, vmax, ticks = create_figures('colorbar_limits_ticks', field_value)
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        m = ax.tripcolor(triang, value_plot, shading='gouraud', cmap=cfg.colormap, vmin=vmin, vmax=vmax)
        create_figures('add_minmax_colorbar', fig, ax, m, label, ticks)
        create_figures('draw_reference_outline', ax, cfg, linestyle='k--')
        create_figures('draw_deformed_boundary_edges', ax, deformed, boundary_edges)
        create_figures('set_equal_axis', ax)
        ax.set_title(label if title is None else title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return
    if operation == 'save_displacement_components':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        u_field = args[2] if len(args) > 2 else kwargs['u_field']
        boundary_edges = args[3] if len(args) > 3 else kwargs['boundary_edges']
        cfg = args[4] if len(args) > 4 else kwargs['cfg']
        save_path = args[5] if len(args) > 5 else kwargs['save_path']
        method_title = args[6] if len(args) > 6 else kwargs['method_title']
        if not cfg.save_figures:
            return
        deformed = nodes + cfg.deformation_scale * u_field
        triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], elements)
        umag = np.linalg.norm(u_field, axis=1)
        values = [(u_field[:, 0], '$u_x$'), (u_field[:, 1], '$u_y$'), (umag, '$|\\mathbf{u}|$')]
        fig, axes = plt.subplots(1, 3, figsize=cfg.component_disp_figsize)
        for ax, (value, label) in zip(axes, values):
            value_plot, vmin, vmax, ticks = create_figures('colorbar_limits_ticks', value)
            m = ax.tripcolor(triang, value_plot, shading='gouraud', cmap=cfg.colormap, vmin=vmin, vmax=vmax)
            create_figures('draw_deformed_boundary_edges', ax, deformed, boundary_edges, linewidth=0.65)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{label}\nmin={vmin:.4f}, max={vmax:.4f}')
            create_figures('add_minmax_colorbar', fig, ax, m, label, ticks, shrink=0.82)
        fig.suptitle(method_title, y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    if operation == 'save_stress_components':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        u_field = args[2] if len(args) > 2 else kwargs['u_field']
        stress_nodal = args[3] if len(args) > 3 else kwargs['stress_nodal']
        vm_nodal = args[4] if len(args) > 4 else kwargs['vm_nodal']
        boundary_edges = args[5] if len(args) > 5 else kwargs['boundary_edges']
        cfg = args[6] if len(args) > 6 else kwargs['cfg']
        save_path = args[7] if len(args) > 7 else kwargs['save_path']
        method_title = args[8] if len(args) > 8 else kwargs['method_title']
        if not cfg.save_figures:
            return
        deformed = nodes + cfg.deformation_scale * u_field
        triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], elements)
        values = [(stress_nodal[:, 0], '$\\sigma_{xx}$'), (stress_nodal[:, 1], '$\\sigma_{yy}$'), (stress_nodal[:, 2], '$\\tau_{xy}$'), (vm_nodal, '$\\sigma_{vM}$')]
        fig, axes = plt.subplots(2, 2, figsize=cfg.component_stress_figsize)
        for ax, (value, label) in zip(axes.ravel(), values):
            value_plot, vmin, vmax, ticks = create_figures('colorbar_limits_ticks', value)
            m = ax.tripcolor(triang, value_plot, shading='gouraud', cmap=cfg.colormap, vmin=vmin, vmax=vmax)
            create_figures('draw_deformed_boundary_edges', ax, deformed, boundary_edges, linewidth=0.65)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{label}\nmin={vmin:.4f}, max={vmax:.4f}')
            create_figures('add_minmax_colorbar', fig, ax, m, label, ticks, shrink=0.82)
        fig.suptitle(method_title, y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    if operation == 'save_deformed_mesh':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        u_field = args[2] if len(args) > 2 else kwargs['u_field']
        cfg = args[3] if len(args) > 3 else kwargs['cfg']
        save_path = args[4] if len(args) > 4 else kwargs['save_path']
        title = args[5] if len(args) > 5 else kwargs['title']
        if not cfg.save_figures:
            return
        deformed = nodes + cfg.deformation_scale * u_field
        fig, ax = plt.subplots(figsize=cfg.field_figsize)
        for conn in elements:
            xy0 = np.vstack([nodes[conn], nodes[conn][0]])
            xyd = np.vstack([deformed[conn], deformed[conn][0]])
            ax.plot(xy0[:, 0], xy0[:, 1], color='0.82', lw=0.25)
            ax.plot(xyd[:, 0], xyd[:, 1], color='#C44E52', lw=0.38)
        create_figures('set_equal_axis', ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return
    if operation == 'profile_filename_tag':
        value = args[0] if args else kwargs['value']
        text = f'{value:.8g}'
        return text.replace('-', 'm').replace('.', 'p')
        return
    if operation == 'make_vertical_profile_points':
        cfg = args[0] if args else kwargs['cfg']
        y_all = np.linspace(0.0, cfg.Ly, cfg.profile_n, dtype=np.float64)
        x_all = np.full_like(y_all, cfg.profile_x)
        tol = 1e-10 * max(cfg.Lx, cfg.Ly, 1.0)
        valid = np.ones(y_all.size, dtype=bool)
        for hole in cfg.holes:
            dist2 = (x_all - float(hole['cx'])) ** 2 + (y_all - float(hole['cy'])) ** 2
            valid &= dist2 > (float(hole['R']) + tol) ** 2
        valid &= (x_all >= -tol) & (x_all <= cfg.Lx + tol)
        valid &= (y_all >= -tol) & (y_all <= cfg.Ly + tol)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            raise RuntimeError(f'Đường profile x={cfg.profile_x:g} không có điểm nào trong miền vật liệu.')
        points = np.column_stack([x_all[valid], y_all[valid]])
        y_profile = y_all[valid]
        segment_id = np.zeros(valid_ids.size, dtype=np.int64)
        if valid_ids.size > 1:
            segment_id[1:] = np.cumsum(np.diff(valid_ids) > 1)
        return (points, y_profile, segment_id)
        return
    if operation == 'interpolate_fem_nodal_field_at_points':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        nodal_field = args[2] if len(args) > 2 else kwargs['nodal_field']
        points = args[3] if len(args) > 3 else kwargs['points']
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
        nodal_field = np.asarray(nodal_field, dtype=np.float64)
        if nodal_field.ndim == 1:
            return np.asarray(np.ma.filled(mtri.LinearTriInterpolator(triang, nodal_field)(points[:, 0], points[:, 1]), np.nan), dtype=np.float64)
        if nodal_field.ndim == 2:
            return np.column_stack([np.asarray(np.ma.filled(mtri.LinearTriInterpolator(triang, nodal_field[:, j])(points[:, 0], points[:, 1]), np.nan), dtype=np.float64) for j in range(nodal_field.shape[1])])
        raise ValueError('nodal_field phải là mảng 1D hoặc 2D.')
        return
    if operation == 'plot_profile_segments':
        ax = args[0] if args else kwargs['ax']
        coordinate = args[1] if len(args) > 1 else kwargs['coordinate']
        value = args[2] if len(args) > 2 else kwargs['value']
        segment_id = args[3] if len(args) > 3 else kwargs['segment_id']
        label = kwargs['label']
        color = kwargs['color']
        linestyle = kwargs['linestyle']
        marker = kwargs['marker']
        markevery = kwargs['markevery']
        first_segment = True
        for sid in np.unique(segment_id):
            mask = segment_id == sid
            finite = mask & np.isfinite(coordinate) & np.isfinite(value)
            if np.count_nonzero(finite) == 0:
                continue
            ax.plot(coordinate[finite], value[finite], color=color, linestyle=linestyle, linewidth=1.7, marker=marker, markersize=4.2, markevery=markevery, label=label if first_segment else None)
            first_segment = False
        return
    if operation == 'save_vertical_profile_comparison':
        mesh = args[0] if args else kwargs['mesh']
        q_trpim = args[1] if len(args) > 1 else kwargs['q_trpim']
        u_fem = args[2] if len(args) > 2 else kwargs['u_fem']
        vm_fem_nodal = args[3] if len(args) > 3 else kwargs['vm_fem_nodal']
        cfg = args[4] if len(args) > 4 else kwargs['cfg']
        out = args[5] if len(args) > 5 else kwargs['out']
        points, y_profile, segment_id = create_figures('make_vertical_profile_points', cfg)
        print(f'\nĐang xây dựng profile x={cfg.profile_x:g}: {points.shape[0]} điểm vật liệu, {int(np.max(segment_id)) + 1} đoạn liên tục ...')
        Phi_p, DPhix_p, DPhiy_p = shape.evaluate_shapes('build_shape_matrices', points, mesh.nodes, cfg)
        fields_trpim = material.evaluate_material('evaluate_fields', q_trpim, points, Phi_p, DPhix_p, DPhiy_p, cfg)
        u_trpim_profile = np.column_stack([fields_trpim.u, fields_trpim.v])
        u_fem_profile = create_figures('interpolate_fem_nodal_field_at_points', mesh.nodes, mesh.elements, u_fem, points)
        vm_fem_profile = create_figures('interpolate_fem_nodal_field_at_points', mesh.nodes, mesh.elements, vm_fem_nodal, points)
        umag_trpim = fields_trpim.umag
        umag_fem = np.linalg.norm(u_fem_profile, axis=1)
        vm_trpim = fields_trpim.vm
        n_bad_fem = int(np.count_nonzero(~np.isfinite(umag_fem) | ~np.isfinite(vm_fem_profile)))
        if n_bad_fem:
            print(f'CẢNH BÁO: có {n_bad_fem} điểm profile FEM bị masked/NaN; các điểm này sẽ tự động bị bỏ khi vẽ.')
        tag = create_figures('profile_filename_tag', cfg.profile_x)
        profile_array = np.column_stack([segment_id, points[:, 0], points[:, 1], fields_trpim.u, fields_trpim.v, umag_trpim, u_fem_profile[:, 0], u_fem_profile[:, 1], umag_fem, vm_trpim, vm_fem_profile])
        np.savetxt(out / f'Profile_x_{tag}_NTRPIM_FEM.csv', profile_array, delimiter=',', header='segment_id,x,y,ux_N_TRPIM,uy_N_TRPIM,umag_N_TRPIM,ux_FEM,uy_FEM,umag_FEM,vm_N_TRPIM,vm_FEM', comments='')
        if cfg.save_figures:
            fig = plt.figure(figsize=cfg.profile_figsize)
            gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.35], hspace=0.34, wspace=0.28)
            ax_geometry = fig.add_subplot(gs[0, :])
            create_figures('draw_reference_outline', ax_geometry, cfg, linestyle='k-')
            for sid in np.unique(segment_id):
                mask = segment_id == sid
                ax_geometry.plot(points[mask, 0], points[mask, 1], color='red', linestyle='--', linewidth=1.6)
            ax_geometry.text(cfg.profile_x + 0.02 * cfg.Lx, 0.06 * cfg.Ly, f'$x={cfg.profile_x:g}$', color='red', fontsize=11, rotation=90, va='bottom')
            create_figures('set_equal_axis', ax_geometry)
            ax_geometry.set_title(f'Profile location: $x={cfg.profile_x:g}$')
            ax_geometry.set_xlim(-0.08 * cfg.Lx, 1.08 * cfg.Lx)
            ax_geometry.set_ylim(-0.04 * cfg.Ly, 1.04 * cfg.Ly)
            ax_u = fig.add_subplot(gs[1, 0])
            create_figures('plot_profile_segments', ax_u, y_profile, umag_trpim, segment_id, label='N-TRPIM', color='#1f77b4', linestyle='-', marker='o', markevery=cfg.profile_markevery)
            create_figures('plot_profile_segments', ax_u, y_profile, umag_fem, segment_id, label='FEM', color='#d62728', linestyle='--', marker='s', markevery=cfg.profile_markevery)
            ax_u.set_xlabel('$y$')
            ax_u.set_ylabel('$|\\mathbf{u}|$')
            ax_u.set_title(f'Displacement magnitude along $x={cfg.profile_x:g}$')
            ax_u.grid(True, linestyle=':', linewidth=0.7, alpha=0.65)
            ax_u.legend(frameon=True)
            ax_u.set_xlim(0.0, cfg.Ly)
            ax_vm = fig.add_subplot(gs[1, 1])
            create_figures('plot_profile_segments', ax_vm, y_profile, vm_trpim, segment_id, label='N-TRPIM', color='#1f77b4', linestyle='-', marker='o', markevery=cfg.profile_markevery)
            create_figures('plot_profile_segments', ax_vm, y_profile, vm_fem_profile, segment_id, label='FEM', color='#d62728', linestyle='--', marker='s', markevery=cfg.profile_markevery)
            ax_vm.set_xlabel('$y$')
            ax_vm.set_ylabel('$\\sigma_{vM}$')
            ax_vm.set_title(f'von Mises stress along $x={cfg.profile_x:g}$')
            ax_vm.grid(True, linestyle=':', linewidth=0.7, alpha=0.65)
            ax_vm.legend(frameon=True)
            ax_vm.set_xlim(0.0, cfg.Ly)
            fig.savefig(out / f'Fig08_Profile_x_{tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.04)
            plt.close(fig)
        print(f'Đã lưu profile: Fig08_Profile_x_{tag}.png và Profile_x_{tag}_NTRPIM_FEM.csv')
        return {'points': points, 'y': y_profile, 'segment_id': segment_id, 'u_trpim': u_trpim_profile, 'u_fem': u_fem_profile, 'umag_trpim': umag_trpim, 'umag_fem': umag_fem, 'vm_trpim': vm_trpim, 'vm_fem': vm_fem_profile}
        return
    if operation == 'save_boundary_displacement_check':
        nodes = args[0] if args else kwargs['nodes']
        boundary_nodes = args[1] if len(args) > 1 else kwargs['boundary_nodes']
        u_trpim = args[2] if len(args) > 2 else kwargs['u_trpim']
        u_fem = args[3] if len(args) > 3 else kwargs['u_fem']
        cfg = args[4] if len(args) > 4 else kwargs['cfg']
        out = args[5] if len(args) > 5 else kwargs['out']
        if not cfg.save_figures:
            return
        top = boundary_nodes['top']
        bottom = boundary_nodes['bottom']
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        if top.size:
            order = np.argsort(nodes[top, 0])
            ids = top[order]
            ax.plot(nodes[ids, 0], u_trpim[ids, 1], 'o-', label='ETRPIM top uy')
            ax.plot(nodes[ids, 0], u_fem[ids, 1], 's--', label='FEM top uy')
        if bottom.size:
            order = np.argsort(nodes[bottom, 0])
            ids = bottom[order]
            ax.plot(nodes[ids, 0], u_trpim[ids, 1], '.-', label='ETRPIM bottom uy')
            ax.plot(nodes[ids, 0], u_fem[ids, 1], 'x--', label='FEM bottom uy')
        ax.axhline(cfg.prescribed_uy_top, color='black', ls=':', label='Prescribed top uy')
        ax.set_xlabel('x')
        ax.set_ylabel('u_y')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'Fig07_boundary_displacement_check.png', dpi=300)
        plt.close(fig)
        return
    raise ValueError(f'Unknown operation in figures: {operation}')
