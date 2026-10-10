# Results, error metrics, and exports
from __future__ import annotations
from .common import *
from . import common
from . import fem
from . import figures
from . import geometry
from . import material
from . import shape
from . import workflow

def export_results(operation, *args, **kwargs):
    if operation == 'fields_to_array':
        x = args[0] if args else kwargs['x']
        f = args[1] if len(args) > 1 else kwargs['f']
        arr = np.column_stack([x[:, 0], x[:, 1], f.x_def[:, 0], f.x_def[:, 1], f.u, f.v, f.umag, f.ux, f.uy, f.vx, f.vy, f.J, f.sxx, f.sxy, f.syy, f.vm])
        header = 'x,y,x_deformed,y_deformed,u_x,u_y,u_mag,duxdx,duxdy,duydx,duydy,J,sxx,sxy,syy,vonMises'
        return (arr, header)
        return
    if operation == 'evaluate_error_metrics':
        u_trpim = args[0] if args else kwargs['u_trpim']
        u_fem = args[1] if len(args) > 1 else kwargs['u_fem']
        stress_trpim = args[2] if len(args) > 2 else kwargs['stress_trpim']
        stress_fem = args[3] if len(args) > 3 else kwargs['stress_fem']
        vm_trpim = args[4] if len(args) > 4 else kwargs['vm_trpim']
        vm_fem = args[5] if len(args) > 5 else kwargs['vm_fem']
        umag_tr = np.linalg.norm(u_trpim, axis=1)
        umag_fe = np.linalg.norm(u_fem, axis=1)
        return {'ux_rel_l2': geometry.prepare_geometry('relative_l2_error', u_trpim[:, 0], u_fem[:, 0]), 'uy_rel_l2': geometry.prepare_geometry('relative_l2_error', u_trpim[:, 1], u_fem[:, 1]), 'umag_rel_l2': geometry.prepare_geometry('relative_l2_error', umag_tr, umag_fe), 'sxx_rel_l2': geometry.prepare_geometry('relative_l2_error', stress_trpim[:, 0], stress_fem[:, 0]), 'syy_rel_l2': geometry.prepare_geometry('relative_l2_error', stress_trpim[:, 1], stress_fem[:, 1]), 'sxy_rel_l2': geometry.prepare_geometry('relative_l2_error', stress_trpim[:, 2], stress_fem[:, 2]), 'vm_rel_l2': geometry.prepare_geometry('relative_l2_error', vm_trpim, vm_fem), 'ux_max_abs': geometry.prepare_geometry('max_abs_error', u_trpim[:, 0], u_fem[:, 0]), 'uy_max_abs': geometry.prepare_geometry('max_abs_error', u_trpim[:, 1], u_fem[:, 1]), 'umag_max_abs': geometry.prepare_geometry('max_abs_error', umag_tr, umag_fe), 'vm_max_abs': geometry.prepare_geometry('max_abs_error', vm_trpim, vm_fem)}
        return
    if operation == 'export_combined_vtu':
        save_path = args[0] if args else kwargs['save_path']
        nodes = args[1] if len(args) > 1 else kwargs['nodes']
        elements = args[2] if len(args) > 2 else kwargs['elements']
        u_trpim = args[3] if len(args) > 3 else kwargs['u_trpim']
        u_fem = args[4] if len(args) > 4 else kwargs['u_fem']
        stress_trpim = args[5] if len(args) > 5 else kwargs['stress_trpim']
        stress_fem = args[6] if len(args) > 6 else kwargs['stress_fem']
        vm_trpim = args[7] if len(args) > 7 else kwargs['vm_trpim']
        vm_fem = args[8] if len(args) > 8 else kwargs['vm_fem']
        J_trpim = args[9] if len(args) > 9 else kwargs['J_trpim']
        J_fem = args[10] if len(args) > 10 else kwargs['J_fem']
        geometry.prepare_geometry('require_meshio')
        points = np.zeros((nodes.shape[0], 3), dtype=np.float64)
        points[:, :2] = nodes
        zeros = np.zeros(nodes.shape[0], dtype=np.float64)
        point_data = {'u_ETRPIM': np.column_stack([u_trpim, zeros]), 'u_FEM': np.column_stack([u_fem, zeros]), 'u_error': np.column_stack([u_trpim - u_fem, zeros]), 'umag_ETRPIM': np.linalg.norm(u_trpim, axis=1), 'umag_FEM': np.linalg.norm(u_fem, axis=1), 'sxx_ETRPIM': stress_trpim[:, 0], 'syy_ETRPIM': stress_trpim[:, 1], 'sxy_ETRPIM': stress_trpim[:, 2], 'vm_ETRPIM': vm_trpim, 'J_ETRPIM': J_trpim, 'sxx_FEM': stress_fem[:, 0], 'syy_FEM': stress_fem[:, 1], 'sxy_FEM': stress_fem[:, 2], 'vm_FEM': vm_fem, 'J_FEM': J_fem, 'vm_abs_error': np.abs(vm_trpim - vm_fem)}
        mesh = meshio.Mesh(points=points, cells=[('triangle', elements)], point_data=point_data)
        meshio.write(str(save_path), mesh)
        print(f'Đã xuất VTU: {save_path}')
        return
    if operation == 'save_summary_table':
        summary = args[0] if args else kwargs['summary']
        out = args[1] if len(args) > 1 else kwargs['out']
        with open(out / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            for key, value in summary.items():
                writer.writerow([key, value])
        with open(out / 'summary.txt', 'w', encoding='utf-8') as f:
            for key, value in summary.items():
                f.write(f'{key}: {value}\n')
        return
    if operation == 'save_all_outputs':
        cfg = args[0] if args else kwargs['cfg']
        mesh = args[1] if len(args) > 1 else kwargs['mesh']
        data = args[2] if len(args) > 2 else kwargs['data']
        ops = args[3] if len(args) > 3 else kwargs['ops']
        trpim_solution = args[4] if len(args) > 4 else kwargs['trpim_solution']
        fem_solution = args[5] if len(args) > 5 else kwargs['fem_solution']
        out = args[6] if len(args) > 6 else kwargs['out']
        q = trpim_solution['q']
        print('\nĐang xây dựng ma trận TRPIM tại nút visualization ...')
        Phi_nodes, DPhix_nodes, DPhiy_nodes = shape.evaluate_shapes('build_shape_matrices', mesh.nodes, mesh.nodes, cfg)
        fields_nodes = material.evaluate_material('evaluate_fields', q, mesh.nodes, Phi_nodes, DPhix_nodes, DPhiy_nodes, cfg)
        u_trpim = np.column_stack([fields_nodes.u, fields_nodes.v])
        stress_trpim = np.column_stack([fields_nodes.sxx, fields_nodes.syy, fields_nodes.sxy])
        u_fem = fem_solution['u']
        stress_fem = geometry.prepare_geometry('element_to_nodal_field', mesh.elements, fem_solution['sigma_elem'], mesh.nodes.shape[0])
        vm_fem = geometry.prepare_geometry('element_to_nodal_field', mesh.elements, fem_solution['vm_elem'], mesh.nodes.shape[0])
        J_fem = geometry.prepare_geometry('element_to_nodal_field', mesh.elements, fem_solution['J_elem'], mesh.nodes.shape[0])
        figures.create_figures('save_fig01_trpim_nodes_samples', data, cfg, out)
        figures.create_figures('save_fig01_fem_mesh', mesh, cfg, out)
        figures.create_figures('save_fig01_boundary_classification', mesh, cfg, out)
        figures.create_figures('save_load_displacement_curves', trpim_solution['step_history'], fem_solution['step_logs'], out, cfg)
        figures.create_figures('save_training_history', trpim_solution['history'], fem_solution['step_logs'], cfg, out)
        figures.create_figures('save_displacement_components', mesh.nodes, mesh.elements, u_trpim, mesh.boundary_edges, cfg, out / 'Fig05_1_ETRPIM.png', 'ETRPIM displacement components')
        figures.create_figures('save_displacement_components', mesh.nodes, mesh.elements, u_fem, mesh.boundary_edges, cfg, out / 'Fig05_2_FEM.png', 'FEM displacement components')
        u_error_abs = np.abs(u_trpim - u_fem)
        umag_error_abs = np.abs(np.linalg.norm(u_trpim, axis=1) - np.linalg.norm(u_fem, axis=1))
        deformed_fem = mesh.nodes + cfg.deformation_scale * u_fem
        triang_fem = mtri.Triangulation(deformed_fem[:, 0], deformed_fem[:, 1], mesh.elements)
        fig, axes = plt.subplots(1, 3, figsize=cfg.component_disp_figsize)
        for ax, (value, label) in zip(axes, [(u_error_abs[:, 0], '$|u_x-u_x^{FEM}|$'), (u_error_abs[:, 1], '$|u_y-u_y^{FEM}|$'), (umag_error_abs, '$||u|-|u|^{FEM}|$')]):
            vp, vmin, vmax, ticks = figures.create_figures('colorbar_limits_ticks', value)
            m = ax.tripcolor(triang_fem, vp, shading='gouraud', cmap=cfg.colormap, vmin=vmin, vmax=vmax)
            figures.create_figures('draw_deformed_boundary_edges', ax, deformed_fem, mesh.boundary_edges, linewidth=0.65)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{label}\nmin={vmin:.4f}, max={vmax:.4f}')
            figures.create_figures('add_minmax_colorbar', fig, ax, m, label, ticks, shrink=0.82)
        fig.suptitle('Absolute displacement error: ETRPIM vs FEM', y=1.01)
        fig.tight_layout()
        fig.savefig(out / 'Fig05_3_absolute_error.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        figures.create_figures('save_stress_components', mesh.nodes, mesh.elements, u_trpim, stress_trpim, fields_nodes.vm, mesh.boundary_edges, cfg, out / 'Fig06_1_ETRPIM.png', 'ETRPIM stress components')
        figures.create_figures('save_stress_components', mesh.nodes, mesh.elements, u_fem, stress_fem, vm_fem, mesh.boundary_edges, cfg, out / 'Fig06_2_FEM.png', 'FEM stress components')
        stress_error = np.abs(stress_trpim - stress_fem)
        vm_error = np.abs(fields_nodes.vm - vm_fem)
        figures.create_figures('save_stress_components', mesh.nodes, mesh.elements, u_fem, stress_error, vm_error, mesh.boundary_edges, cfg, out / 'Fig06_3_absolute_error.png', 'Absolute stress error: ETRPIM vs FEM')
        figures.create_figures('save_boundary_displacement_check', mesh.nodes, mesh.boundary_nodes, u_trpim, u_fem, cfg, out)
        profile_results = figures.create_figures('save_vertical_profile_comparison', mesh=mesh, q_trpim=q, u_fem=u_fem, vm_fem_nodal=vm_fem, cfg=cfg, out=out)
        trpim_fields = [(fields_nodes.u, 'ETRPIM_Deformed_01_u_x.png', '$u_x$'), (fields_nodes.v, 'ETRPIM_Deformed_02_u_y.png', '$u_y$'), (fields_nodes.umag, 'ETRPIM_Deformed_03_u_magnitude.png', '$|\\mathbf{u}|$'), (fields_nodes.sxx, 'ETRPIM_Deformed_04_sigma_xx.png', '$\\sigma_{xx}$'), (fields_nodes.syy, 'ETRPIM_Deformed_05_sigma_yy.png', '$\\sigma_{yy}$'), (fields_nodes.sxy, 'ETRPIM_Deformed_06_tau_xy.png', '$\\tau_{xy}$'), (fields_nodes.vm, 'ETRPIM_Deformed_07_vonMises.png', '$\\sigma_{vM}$'), (fields_nodes.J, 'ETRPIM_Deformed_08_J.png', '$J$')]
        for value, name, label in trpim_fields:
            figures.create_figures('plot_field_on_deformed', mesh.nodes, mesh.elements, u_trpim, value, mesh.boundary_edges, cfg, out / name, label)
        umag_fem = np.linalg.norm(u_fem, axis=1)
        fem_fields = [(u_fem[:, 0], 'FEM_Deformed_01_u_x.png', '$u_x$'), (u_fem[:, 1], 'FEM_Deformed_02_u_y.png', '$u_y$'), (umag_fem, 'FEM_Deformed_03_u_magnitude.png', '$|\\mathbf{u}|$'), (stress_fem[:, 0], 'FEM_Deformed_04_sigma_xx.png', '$\\sigma_{xx}$'), (stress_fem[:, 1], 'FEM_Deformed_05_sigma_yy.png', '$\\sigma_{yy}$'), (stress_fem[:, 2], 'FEM_Deformed_06_tau_xy.png', '$\\tau_{xy}$'), (vm_fem, 'FEM_Deformed_07_vonMises.png', '$\\sigma_{vM}$'), (J_fem, 'FEM_Deformed_08_J.png', '$J$')]
        for value, name, label in fem_fields:
            figures.create_figures('plot_field_on_deformed', mesh.nodes, mesh.elements, u_fem, value, mesh.boundary_edges, cfg, out / name, label)
        figures.create_figures('save_deformed_mesh', mesh.nodes, mesh.elements, u_trpim, cfg, out / 'ETRPIM_deformed_mesh.png', 'ETRPIM deformed mesh')
        figures.create_figures('save_deformed_mesh', mesh.nodes, mesh.elements, u_fem, cfg, out / 'FEM_deformed_mesh.png', 'FEM deformed mesh')
        fields_int = material.evaluate_material('evaluate_fields', q, data.x_int, ops.Phi_int, ops.DPhix_int, ops.DPhiy_int, cfg)
        arr, header = export_results('fields_to_array', data.x_int, fields_int)
        np.savetxt(out / 'ETRPIM_sample_point_results.csv', arr, delimiter=',', header=header, comments='')
        arr, header = export_results('fields_to_array', mesh.nodes, fields_nodes)
        np.savetxt(out / 'ETRPIM_nodal_results.csv', arr, delimiter=',', header=header, comments='')
        np.savetxt(out / 'ETRPIM_history.csv', trpim_solution['history'], delimiter=',', header='iter,load_step,load_factor,target_top_uy,loss,E_in,L_bottom,L_top,minJ_sample,mean_top_uy,grad_norm,minJ_guard,min_area_ratio,accepted_scale,effective_lr', comments='')
        np.savetxt(out / 'ETRPIM_step_history.csv', trpim_solution['step_history'], delimiter=',', header='step,load_factor,target_top_uy,mean_top_uy,reaction_y,minJ_sample,loss,local_iterations,minJ_guard,min_area_ratio,rejected_updates', comments='')
        fem_step_array = np.array([[r['step'], r['load_factor'], r['target_uy'], r['reaction_y'], r['res_norm'], r['du_norm'], r['J_min'], r['J_max'], r['newton_iterations']] for r in fem_solution['step_logs']])
        np.savetxt(out / 'FEM_step_history.csv', fem_step_array, delimiter=',', header='step,load_factor,target_uy,reaction_y,res_norm,du_norm,J_min,J_max,newton_iterations', comments='')
        metrics = export_results('evaluate_error_metrics', u_trpim, u_fem, stress_trpim, stress_fem, fields_nodes.vm, vm_fem)
        trpim_energy = float(np.sum(data.w_int * (0.5 * cfg.mu * ((1.0 + fields_int.ux) ** 2 + fields_int.uy ** 2 + fields_int.vx ** 2 + (1.0 + fields_int.vy) ** 2 - 2.0) - cfg.mu * np.log(fields_int.J) + 0.5 * cfg.lam * np.log(fields_int.J) ** 2)))
        fem_energy = fem.solve_fem('compute_fem_internal_energy', u_fem, mesh.elements, fem_solution['gradN_all'], fem_solution['area_all'], cfg)
        relative_energy_error = abs(trpim_energy - fem_energy) / (abs(fem_energy) + 1e-14)
        summary: Dict[str, float | str] = {'case': cfg.case_name, 'material': 'Compressible Neo-Hookean 2D', 'E': cfg.E, 'nu': cfg.nu, 'mu': cfg.mu, 'lambda': cfg.lam, 'prescribed_uy_top': cfg.prescribed_uy_top, 'profile_x': cfg.profile_x, 'profile_points_in_material': int(profile_results['points'].shape[0]), 'profile_segments': int(np.max(profile_results['segment_id'])) + 1, 'num_nodes': int(mesh.nodes.shape[0]), 'num_elements': int(mesh.elements.shape[0]), 'sample_n': int(data.x_int.shape[0]), 'samples_per_node_actual': float(data.x_int.shape[0] / mesh.nodes.shape[0]), 'support_nodes': cfg.support_nodes, 'max_support_nodes': cfg.max_support_nodes, 'trpim_wall_time_s': float(trpim_solution['time']), 'fem_wall_time_s': float(fem_solution['time']), 'trpim_min_J_nodal': float(np.min(fields_nodes.J)), 'trpim_min_J_guard_final': float(trpim_solution['step_history'][-1, 8]), 'trpim_min_area_ratio_final': float(trpim_solution['step_history'][-1, 9]), 'trpim_rejected_updates_total': int(np.sum(trpim_solution['step_history'][:, 10])), 'fem_min_J_nodal': float(np.min(J_fem)), 'trpim_internal_energy': trpim_energy, 'fem_internal_energy': fem_energy, 'relative_energy_error': relative_energy_error, **metrics}
        export_results('save_summary_table', summary, out)
        with open(out / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(workflow.execute('to_jsonable_config', cfg), f, indent=2, ensure_ascii=False)
        np.savez_compressed(out / 'solution_ETRPIM_and_FEM.npz', nodes=mesh.nodes, elements=mesh.elements, q_trpim=q, u_trpim=u_trpim, stress_trpim=stress_trpim, vm_trpim=fields_nodes.vm, J_trpim=fields_nodes.J, u_fem=u_fem, stress_fem_nodal=stress_fem, vm_fem_nodal=vm_fem, J_fem_nodal=J_fem, sigma_fem_element=fem_solution['sigma_elem'], vm_fem_element=fem_solution['vm_elem'], J_fem_element=fem_solution['J_elem'], x_sample=data.x_int, w_sample=data.w_int, trpim_history=trpim_solution['history'], trpim_step_history=trpim_solution['step_history'], fem_step_history=fem_step_array, profile_x=np.asarray([cfg.profile_x], dtype=np.float64), profile_points=profile_results['points'], profile_y=profile_results['y'], profile_segment_id=profile_results['segment_id'], profile_u_trpim=profile_results['u_trpim'], profile_u_fem=profile_results['u_fem'], profile_umag_trpim=profile_results['umag_trpim'], profile_umag_fem=profile_results['umag_fem'], profile_vm_trpim=profile_results['vm_trpim'], profile_vm_fem=profile_results['vm_fem'])
        export_results('export_combined_vtu', out / 'results_ETRPIM_and_FEM.vtu', mesh.nodes, mesh.elements, u_trpim, u_fem, stress_trpim, stress_fem, fields_nodes.vm, vm_fem, fields_nodes.J, J_fem)
        print('\nSai số cuối cùng ETRPIM so với FEM:')
        for key, value in metrics.items():
            print(f'  {key:18s} = {value:.8e}')
        print(f'  relative_energy_error = {relative_energy_error:.8e}')
        return
    raise ValueError(f'Unknown operation in exports: {operation}')
