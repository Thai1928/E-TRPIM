# Configuration and execution workflow
from __future__ import annotations
from .common import *
from . import common
from . import exports
from . import fem
from . import geometry
from . import shape
from . import trpim

def execute(operation, *args, **kwargs):
    if operation == 'run':
        cfg = args[0] if args else kwargs['cfg']
        script_dir = args[1] if len(args) > 1 else kwargs['script_dir']
        execute('finalize_config', cfg)
        geometry.prepare_geometry('require_meshio')
        out = script_dir / cfg.output_folder
        geometry.prepare_geometry('ensure_dir', out)
        msh_path = out / 'perforated_plate_3_holes.msh'
        print('\n' + '=' * 78)
        print('ETRPIM + FEM | 2D perforated plate | Compressible Neo-Hookean')
        print(f'Hình học: Lx={cfg.Lx}, Ly={cfg.Ly}, 3 lỗ={cfg.holes}')
        print(f'Vật liệu: E={cfg.E:g}, nu={cfg.nu:g}, mu={cfg.mu:g}, lambda={cfg.lam:g}')
        print(f'Chuyển vị biên trên: uy={cfg.prescribed_uy_top:g}')
        print(f'TRPIM: sample_n={cfg.sample_n}, support={cfg.support_nodes}')
        print(f'Adam: lr={cfg.lr:g}, load_steps={cfg.load_steps}, iter/step={cfg.iter_per_load_step}')
        print(f'Output: {out}')
        print('=' * 78)
        if cfg.regenerate_mesh or not msh_path.exists():
            geometry.prepare_geometry('require_gmsh')
            print('\nĐang tạo lưới Gmsh ...')
            geometry.prepare_geometry('generate_gmsh_full_plate_with_3_holes', msh_path, cfg)
        else:
            print(f'\nDùng lại lưới hiện có: {msh_path}')
        mesh = geometry.prepare_geometry('load_gmsh_mesh', msh_path)
        cfg.node_spacing_estimate = geometry.prepare_geometry('estimate_node_spacing', mesh.nodes)
        print(f'Lưới: num_nodes={mesh.nodes.shape[0]}, num_elements={mesh.elements.shape[0]}, node_spacing_estimate={cfg.node_spacing_estimate:.6g}')
        data = geometry.prepare_geometry('make_model_data', mesh, cfg)
        ops = shape.evaluate_shapes('build_operators', data, cfg)
        trpim_solution = trpim.optimize_trpim('solve_etrpim', data, ops, cfg)
        fem_solution = fem.solve_fem('solve_fem_reference', mesh, cfg)
        exports.export_results('save_all_outputs', cfg, mesh, data, ops, trpim_solution, fem_solution, out)
        print(f'\nHoàn tất. Toàn bộ kết quả đã lưu tại:\n  {out}')
        return
    if operation == 'make_quick_config':
        cfg = Config()
        cfg.output_folder = 'OUTPUT_ETRPIM_2D_PERFORATED_PLATE_QUICKTEST'
        cfg.lc_plate = 0.22
        cfg.lc_hole = 0.12
        cfg.sample_n = 100
        cfg.bottom_boundary_samples = 15
        cfg.top_boundary_samples = 15
        cfg.support_nodes = 12
        cfg.max_support_nodes = 24
        cfg.load_steps = 1
        cfg.iter_per_load_step = 5
        cfg.print_every = 1
        cfg.lr = 0.0001
        cfg.max_coeff_update = 0.02
        cfg.prescribed_uy_top = 0.02
        cfg.fem_load_steps = 1
        cfg.fem_newton_max_iter = 12
        cfg.fem_n_jobs = 1
        return cfg
        return
    if operation == 'parse_args':
        parser = argparse.ArgumentParser(description='ETRPIM và FEM cho tấm 2D có 3 lỗ, vật liệu Neo-Hookean nén được.')
        parser.add_argument('--quick', action='store_true', help='Chạy cấu hình kiểm tra nhanh.')
        parser.add_argument('--no-figures', action='store_true', help='Không lưu hình.')
        parser.add_argument('--output', type=str, default=None, help='Tên thư mục output.')
        parser.add_argument('--sample-n', type=int, default=None)
        parser.add_argument('--samples-per-node', type=float, default=None)
        parser.add_argument('--no-auto-sample-density', action='store_true')
        parser.add_argument('--sample-mode', choices=['halton', 'uniform_midpoint'], default=None)
        parser.add_argument('--load-steps', type=int, default=None)
        parser.add_argument('--iterations-per-step', type=int, default=None)
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--alpha-bottom', type=float, default=None)
        parser.add_argument('--alpha-top', type=float, default=None)
        parser.add_argument('--support-nodes', type=int, default=None)
        parser.add_argument('--max-support-nodes', type=int, default=None)
        parser.add_argument('--cheb-order', type=int, choices=[1, 2], default=None)
        parser.add_argument('--top-uy', type=float, default=None)
        parser.add_argument('--lc-plate', type=float, default=None)
        parser.add_argument('--lc-hole', type=float, default=None)
        parser.add_argument('--fem-load-steps', type=int, default=None)
        parser.add_argument('--fem-jobs', type=int, default=None)
        parser.add_argument('--profile-x', type=float, default=None)
        parser.add_argument('--profile-n', type=int, default=None)
        parser.add_argument('--reuse-mesh', action='store_true')
        parser.add_argument('--hard-boundary-nodes', action='store_true')
        parser.add_argument('--soft-boundary-nodes', action='store_true')
        parser.add_argument('--disable-visibility', action='store_true')
        return parser.parse_args()
        return
    if operation == 'main':
        args = execute('parse_args')
        cfg = execute('make_quick_config') if args.quick else Config()
        if args.no_figures:
            cfg.save_figures = False
        if args.output is not None:
            cfg.output_folder = args.output
        if args.sample_n is not None:
            cfg.sample_n = args.sample_n
        if args.samples_per_node is not None:
            cfg.samples_per_node = args.samples_per_node
        if args.no_auto_sample_density:
            cfg.auto_sample_density = False
        if args.sample_mode is not None:
            cfg.sample_mode = args.sample_mode
        if args.load_steps is not None:
            cfg.load_steps = args.load_steps
        if args.iterations_per_step is not None:
            cfg.iter_per_load_step = args.iterations_per_step
        if args.lr is not None:
            cfg.lr = args.lr
        if args.alpha_bottom is not None:
            cfg.alpha_bottom = args.alpha_bottom
        if args.alpha_top is not None:
            cfg.alpha_top = args.alpha_top
        if args.support_nodes is not None:
            cfg.support_nodes = args.support_nodes
        if args.max_support_nodes is not None:
            cfg.max_support_nodes = args.max_support_nodes
        if args.cheb_order is not None:
            cfg.cheb_order = args.cheb_order
        if args.top_uy is not None:
            cfg.prescribed_uy_top = args.top_uy
        if args.lc_plate is not None:
            cfg.lc_plate = args.lc_plate
        if args.lc_hole is not None:
            cfg.lc_hole = args.lc_hole
        if args.fem_load_steps is not None:
            cfg.fem_load_steps = args.fem_load_steps
        if args.fem_jobs is not None:
            cfg.fem_n_jobs = max(1, args.fem_jobs)
        if args.profile_x is not None:
            cfg.profile_x = args.profile_x
        if args.profile_n is not None:
            cfg.profile_n = args.profile_n
        if args.reuse_mesh:
            cfg.regenerate_mesh = False
        if args.hard_boundary_nodes:
            cfg.hard_impose_boundary_nodes = True
        if args.soft_boundary_nodes:
            cfg.hard_impose_boundary_nodes = False
        if args.disable_visibility:
            cfg.use_visibility_criterion = False
        execute('run', cfg, Path(__file__).resolve().parent.parent)
        return
    if operation == 'finalize_config':
        self = args[0] if args else kwargs['self']
        if self.sample_n <= 0:
            self.sample_n = 2500
        if self.bottom_boundary_samples < 2 or self.top_boundary_samples < 2:
            raise ValueError('Số điểm mẫu trên biên phải >= 2.')
        if not 0.0 <= self.nu < 0.5:
            raise ValueError('Với mô hình nén được, cần 0 <= nu < 0.5.')
        if len(self.holes) != 3:
            raise ValueError('Script hiện được thiết lập cho đúng 3 lỗ tròn.')
        if not 0.0 <= self.profile_x <= self.Lx:
            raise ValueError('profile_x phải nằm trong đoạn [0, Lx].')
        if self.profile_n < 2:
            raise ValueError('profile_n phải >= 2.')
        if self.profile_markevery < 1:
            raise ValueError('profile_markevery phải >= 1.')
        return self
        return
    if operation == 'to_jsonable_config':
        self = args[0] if args else kwargs['self']
        d = asdict(self)
        d['mu'] = self.mu
        d['lambda'] = self.lam
        d['domain_area'] = self.domain_area
        d['base_support_radius'] = self.base_support_radius
        d['total_iterations'] = self.total_iterations
        return d
        return
    raise ValueError(f'Unknown operation in workflow: {operation}')
