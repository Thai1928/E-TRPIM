# Geometry and integration-point sampling
from __future__ import annotations
from .common import *
from . import common
from .common import _GMSH_IMPORT_ERROR, _MESHIO_IMPORT_ERROR

def prepare_geometry(operation, *args, **kwargs):
    if operation == 'require_gmsh':
        if gmsh is None:
            detail = f' | Chi tiết: {_GMSH_IMPORT_ERROR}' if _GMSH_IMPORT_ERROR else ''
            raise ImportError("Không nạp được gmsh. Cài Python package bằng 'pip install gmsh' và bảo đảm thư viện hệ thống OpenGL/GLU đã có." + detail)
        return
    if operation == 'require_meshio':
        if meshio is None:
            detail = f' | Chi tiết: {_MESHIO_IMPORT_ERROR}' if _MESHIO_IMPORT_ERROR else ''
            raise ImportError('Không nạp được meshio. Cài bằng: pip install meshio' + detail)
        return
    if operation == 'ensure_dir':
        path = args[0] if args else kwargs['path']
        path.mkdir(parents=True, exist_ok=True)
        return
    if operation == 'relative_l2_error':
        pred = args[0] if args else kwargs['pred']
        ref = args[1] if len(args) > 1 else kwargs['ref']
        eps = args[2] if len(args) > 2 else kwargs.get('eps', 1e-14)
        pred = np.asarray(pred).reshape(-1)
        ref = np.asarray(ref).reshape(-1)
        return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))
        return
    if operation == 'max_abs_error':
        pred = args[0] if args else kwargs['pred']
        ref = args[1] if len(args) > 1 else kwargs['ref']
        return float(np.max(np.abs(np.asarray(pred) - np.asarray(ref))))
        return
    if operation == 'element_to_nodal_field':
        elements = args[0] if args else kwargs['elements']
        elem_field = args[1] if len(args) > 1 else kwargs['elem_field']
        num_nodes = args[2] if len(args) > 2 else kwargs['num_nodes']
        elem_field = np.asarray(elem_field)
        if elem_field.ndim == 1:
            nodal_sum = np.zeros(num_nodes, dtype=float)
        else:
            nodal_sum = np.zeros((num_nodes, elem_field.shape[1]), dtype=float)
        nodal_count = np.zeros(num_nodes, dtype=float)
        for e, conn in enumerate(elements):
            nodal_sum[conn] += elem_field[e]
            nodal_count[conn] += 1.0
        if elem_field.ndim == 1:
            return nodal_sum / np.maximum(nodal_count, 1.0)
        return nodal_sum / np.maximum(nodal_count[:, None], 1.0)
        return
    if operation == 'estimate_node_spacing':
        nodes = args[0] if args else kwargs['nodes']
        tree = cKDTree(nodes)
        d, _ = tree.query(nodes, k=2)
        nearest = d[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        if nearest.size == 0:
            return 1.0
        return float(np.median(nearest))
        return
    if operation == 'generate_gmsh_full_plate_with_3_holes':
        msh_path = args[0] if args else kwargs['msh_path']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        prepare_geometry('require_gmsh')
        gmsh.initialize()
        try:
            gmsh.model.add('perforated_plate_3_holes_etrpim_fem')
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cfg.lc_plate)
            p2 = gmsh.model.geo.addPoint(cfg.Lx, 0.0, 0.0, cfg.lc_plate)
            p3 = gmsh.model.geo.addPoint(cfg.Lx, cfg.Ly, 0.0, cfg.lc_plate)
            p4 = gmsh.model.geo.addPoint(0.0, cfg.Ly, 0.0, cfg.lc_plate)
            l_bottom = gmsh.model.geo.addLine(p1, p2)
            l_right = gmsh.model.geo.addLine(p2, p3)
            l_top = gmsh.model.geo.addLine(p3, p4)
            l_left = gmsh.model.geo.addLine(p4, p1)
            outer_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
            hole_loops = []
            all_hole_arcs = []
            for hole in cfg.holes:
                cx, cy, radius = (hole['cx'], hole['cy'], hole['R'])
                pc = gmsh.model.geo.addPoint(cx, cy, 0.0, cfg.lc_hole)
                pr = gmsh.model.geo.addPoint(cx + radius, cy, 0.0, cfg.lc_hole)
                pt = gmsh.model.geo.addPoint(cx, cy + radius, 0.0, cfg.lc_hole)
                pl = gmsh.model.geo.addPoint(cx - radius, cy, 0.0, cfg.lc_hole)
                pb = gmsh.model.geo.addPoint(cx, cy - radius, 0.0, cfg.lc_hole)
                c1 = gmsh.model.geo.addCircleArc(pr, pc, pt)
                c2 = gmsh.model.geo.addCircleArc(pt, pc, pl)
                c3 = gmsh.model.geo.addCircleArc(pl, pc, pb)
                c4 = gmsh.model.geo.addCircleArc(pb, pc, pr)
                hole_loops.append(gmsh.model.geo.addCurveLoop([c1, c2, c3, c4]))
                all_hole_arcs.extend([c1, c2, c3, c4])
            surface = gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops)
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(1, [l_left], 1)
            gmsh.model.setPhysicalName(1, 1, 'left')
            gmsh.model.addPhysicalGroup(1, [l_right], 2)
            gmsh.model.setPhysicalName(1, 2, 'right')
            gmsh.model.addPhysicalGroup(1, [l_top], 3)
            gmsh.model.setPhysicalName(1, 3, 'top')
            gmsh.model.addPhysicalGroup(1, [l_bottom], 4)
            gmsh.model.setPhysicalName(1, 4, 'bottom')
            gmsh.model.addPhysicalGroup(1, all_hole_arcs, 5)
            gmsh.model.setPhysicalName(1, 5, 'hole')
            gmsh.model.addPhysicalGroup(2, [surface], 10)
            gmsh.model.setPhysicalName(2, 10, 'domain')
            gmsh.option.setNumber('Mesh.Algorithm', 6)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMin', min(cfg.lc_plate, cfg.lc_hole))
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', max(cfg.lc_plate, cfg.lc_hole))
            gmsh.model.mesh.generate(2)
            gmsh.write(str(msh_path))
        finally:
            gmsh.finalize()
        return
    if operation == 'load_gmsh_mesh':
        msh_path = args[0] if args else kwargs['msh_path']
        prepare_geometry('require_meshio')
        mesh = meshio.read(str(msh_path))
        nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
        if 'gmsh:physical' not in mesh.cell_data:
            raise ValueError("Không tìm thấy cell_data 'gmsh:physical' trong lưới Gmsh.")
        physical_data = mesh.cell_data['gmsh:physical']
        triangle_blocks: List[np.ndarray] = []
        line_blocks: List[np.ndarray] = []
        line_tag_blocks: List[np.ndarray] = []
        for i, block in enumerate(mesh.cells):
            tags = np.asarray(physical_data[i])
            if block.type == 'triangle':
                triangle_blocks.append(np.asarray(block.data, dtype=np.int64))
            elif block.type == 'line':
                line_blocks.append(np.asarray(block.data, dtype=np.int64))
                line_tag_blocks.append(np.asarray(tags, dtype=np.int64))
        if not triangle_blocks:
            raise ValueError('Lưới không có phần tử triangle.')
        if not line_blocks:
            raise ValueError('Lưới không có phần tử line cho biên.')
        elements = np.vstack(triangle_blocks).astype(np.int64)
        line_cells = np.vstack(line_blocks).astype(np.int64)
        line_tags = np.concatenate(line_tag_blocks).astype(np.int64)
        boundary_edges: Dict[str, List[List[int]]] = {'left': [], 'right': [], 'top': [], 'bottom': [], 'hole': []}
        tag_to_name = {1: 'left', 2: 'right', 3: 'top', 4: 'bottom', 5: 'hole'}
        for edge, tag in zip(line_cells, line_tags):
            if int(tag) in tag_to_name:
                boundary_edges[tag_to_name[int(tag)]].append(edge.tolist())
        boundary_edges_np: Dict[str, np.ndarray] = {}
        boundary_nodes: Dict[str, np.ndarray] = {}
        for name, edges in boundary_edges.items():
            if edges:
                arr = np.asarray(edges, dtype=np.int64)
                boundary_edges_np[name] = arr
                boundary_nodes[name] = np.unique(arr.reshape(-1))
            else:
                boundary_edges_np[name] = np.empty((0, 2), dtype=np.int64)
                boundary_nodes[name] = np.empty(0, dtype=np.int64)
        return MeshData(nodes, elements, boundary_edges_np, boundary_nodes)
        return
    if operation == 'vdc_value':
        index = args[0] if args else kwargs['index']
        base = args[1] if len(args) > 1 else kwargs['base']
        f = 1.0 / base
        r = 0.0
        k = index
        while k > 0:
            r += f * (k % base)
            k //= base
            f /= base
        return r
        return
    if operation == 'point_is_inside_domain':
        x = args[0] if args else kwargs['x']
        y = args[1] if len(args) > 1 else kwargs['y']
        cfg = args[2] if len(args) > 2 else kwargs['cfg']
        if not (0.0 <= x <= cfg.Lx and 0.0 <= y <= cfg.Ly):
            return False
        for hole in cfg.holes:
            if (x - hole['cx']) ** 2 + (y - hole['cy']) ** 2 < hole['R'] ** 2:
                return False
        return True
        return
    if operation == 'make_halton_domain_points':
        n = args[0] if args else kwargs['n']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        points: List[Tuple[float, float]] = []
        index = 1
        max_trials = max(1000, 20 * n)
        while len(points) < n and index <= max_trials:
            x = cfg.Lx * prepare_geometry('vdc_value', index, 2)
            y = cfg.Ly * prepare_geometry('vdc_value', index, 3)
            if prepare_geometry('point_is_inside_domain', x, y, cfg):
                points.append((x, y))
            index += 1
        if len(points) < n:
            raise RuntimeError(f'Không sinh đủ {n} điểm Halton trong miền; chỉ có {len(points)} điểm.')
        return np.asarray(points, dtype=np.float64)
        return
    if operation == 'make_uniform_rejection_points':
        n = args[0] if args else kwargs['n']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        ratio = cfg.Lx / cfg.Ly
        nx = max(2, int(math.ceil(math.sqrt(n * ratio))))
        ny = max(2, int(math.ceil(n / nx)))
        while True:
            xv = (np.arange(nx) + 0.5) * cfg.Lx / nx
            yv = (np.arange(ny) + 0.5) * cfg.Ly / ny
            X, Y = np.meshgrid(xv, yv, indexing='xy')
            candidate = np.column_stack((X.ravel(), Y.ravel()))
            keep = np.array([prepare_geometry('point_is_inside_domain', x, y, cfg) for x, y in candidate])
            valid = candidate[keep]
            if valid.shape[0] >= n:
                return valid
            nx = int(math.ceil(nx * 1.15)) + 1
            ny = int(math.ceil(ny * 1.15)) + 1
        return
    if operation == 'make_interior_sample_points':
        cfg = args[0] if args else kwargs['cfg']
        if cfg.sample_mode.lower() == 'halton':
            x = prepare_geometry('make_halton_domain_points', cfg.sample_n, cfg)
        elif cfg.sample_mode.lower() == 'uniform_midpoint':
            x = prepare_geometry('make_uniform_rejection_points', cfg.sample_n, cfg)
        else:
            raise ValueError("sample_mode phải là 'halton' hoặc 'uniform_midpoint'.")
        w = np.full(x.shape[0], cfg.domain_area * cfg.thickness / x.shape[0], dtype=np.float64)
        return (x, w)
        return
    if operation == 'make_line_samples':
        p0 = args[0] if args else kwargs['p0']
        p1 = args[1] if len(args) > 1 else kwargs['p1']
        n = args[2] if len(args) > 2 else kwargs['n']
        thickness = args[3] if len(args) > 3 else kwargs['thickness']
        a = np.linspace(0.0, 1.0, n)
        x = p0 + a[:, None] * (p1 - p0)
        length = float(np.linalg.norm(p1 - p0))
        ds = length / (n - 1)
        w = np.full(n, ds * thickness, dtype=np.float64)
        w[0] *= 0.5
        w[-1] *= 0.5
        return (x, w)
        return
    if operation == 'make_guard_points':
        nodes = args[0] if args else kwargs['nodes']
        elements = args[1] if len(args) > 1 else kwargs['elements']
        return np.mean(nodes[elements], axis=1)
        return
    if operation == 'make_model_data':
        mesh = args[0] if args else kwargs['mesh']
        cfg = args[1] if len(args) > 1 else kwargs['cfg']
        if cfg.auto_sample_density:
            auto_n = int(math.ceil(cfg.samples_per_node * mesh.nodes.shape[0]))
            if auto_n > cfg.sample_n:
                print(f'Tự tăng sample_n: {cfg.sample_n} -> {auto_n} ({cfg.samples_per_node:g} samples/node)')
                cfg.sample_n = auto_n
        x_int, w_int = prepare_geometry('make_interior_sample_points', cfg)
        x_bottom, w_bottom = prepare_geometry('make_line_samples', np.array([0.0, 0.0]), np.array([cfg.Lx, 0.0]), cfg.bottom_boundary_samples, cfg.thickness)
        x_top, w_top = prepare_geometry('make_line_samples', np.array([0.0, cfg.Ly]), np.array([cfg.Lx, cfg.Ly]), cfg.top_boundary_samples, cfg.thickness)
        x_guard = prepare_geometry('make_guard_points', mesh.nodes, mesh.elements)
        return ModelData(nodes=mesh.nodes, elements=mesh.elements, fixed_bottom_nodes=mesh.boundary_nodes['bottom'], top_nodes=mesh.boundary_nodes['top'], x_int=x_int, w_int=w_int, x_bottom=x_bottom, w_bottom=w_bottom, x_top=x_top, w_top=w_top, x_guard=x_guard, boundary_edges=mesh.boundary_edges, boundary_nodes=mesh.boundary_nodes)
        return
    raise ValueError(f'Unknown operation in geometry: {operation}')
