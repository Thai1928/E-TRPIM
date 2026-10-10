import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = Path("ZZZ_Visualization_Ex1")
MAIN_RESULTS_PATH = RESULT_DIR / "solution_ETRPIM_and_FEM_Regular.npz"
NTRPIM2_RESULTS_PATH = RESULT_DIR / "solution_ETRPIM_and_FEM_Irregular.npz"
OUTPUT_DIR = BASE_DIR if BASE_DIR.name == RESULT_DIR.name else BASE_DIR / RESULT_DIR
OUTPUT_NAME = "Fig02_3_FEM_NTRPIM1_NTRPIM2_load_reaction"

DPI = 1000
FIGSIZE = (7.0, 5.0)
X_LABEL = "Prescribed top displacement"
Y_LABEL = "Vertical reaction"
ADD_ZERO_POINT = False
USE_ABSOLUTE_REACTION = False
X_LIMITS = None
Y_LIMITS = None

LEGEND_LOC = "upper left"
LEGEND_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 11
GRID_ALPHA = 0.55
GRID_LINEWIDTH = 0.8

CURVES = [
    {"label": "FEM", "data": "fem_saved", "color": "black", "linestyle": "-",
     "marker": "s", "linewidth": 2.0, "markersize": 5,
     "markerfacecolor": "black", "markeredgecolor": "black"},
    {"label": "E-TRPIM (Regular)", "data": "ntrpim_saved", "color": "red", "linestyle": "-",
     "marker": "^", "linewidth": 2.0, "markersize": 5,
     "markerfacecolor": "red", "markeredgecolor": "red"},
    {"label": "E-TRPIM (Irregular)", "data": "ntrpim_external", "path": NTRPIM2_RESULTS_PATH,
     "color": "blue", "linestyle": "-", "marker": "o", "linewidth": 2.0,
     "markersize": 4, "markerfacecolor": "blue", "markeredgecolor": "blue",
     "markeredgewidth": 1.4},
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})


def find_result_file(path):
    path = Path(path)
    locations = [path] if path.is_absolute() else [BASE_DIR / path, Path.cwd() / path, BASE_DIR.parent / path, BASE_DIR / path.name]
    filenames = ["solution_ETRPIM_and_FEM.npz", "ETRPIM_step_history.csv",
                 "FEM_step_history.csv", "step_history.csv", "load_reaction.csv", "load_displacement.csv"]

    for location in locations:
        if location.is_file():
            return location
        if location.is_dir():
            for name in filenames:
                if (location / name).is_file():
                    return location / name

    raise FileNotFoundError(f"Result file not found: {path}\nChecked: " + ", ".join(map(str, locations)))


def prepare_curve(x, y, label):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"{label}: x and y have different lengths ({len(x)} and {len(y)}).")

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        raise ValueError(f"{label}: no valid data points found.")
    if USE_ABSOLUTE_REACTION:
        y = np.abs(y)

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.r_[x[1:] != x[:-1], True]
    x, y = x[keep], y[keep]

    if ADD_ZERO_POINT:
        if np.isclose(x[0], 0):
            x[0], y[0] = 0.0, 0.0
        else:
            x, y = np.r_[0.0, x], np.r_[0.0, y]

    return x, y


def load_npz(path, source):
    # FEM: columns 2/3; E-TRPIM: columns 2/4 (displacement/reaction).
    key, x_col, y_col = ("fem_step_history", 2, 3) if source == "fem" else ("trpim_step_history", 2, 4)
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{path}: missing '{key}'. Available keys: {data.files}")
        history = np.asarray(data[key], dtype=float)

    if history.ndim != 2 or history.shape[1] <= y_col:
        raise ValueError(f"{path}: invalid shape for {key}: {history.shape}")
    return history[:, x_col], history[:, y_col]


def load_csv(path, source):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header.")

        columns = {name.strip().lower(): name for name in reader.fieldnames}
        displacement_names = ("target_uy", "target_top_uy") if source == "fem" else ("target_top_uy", "target_uy")
        displacement_names += ("prescribed_top_displacement", "prescribed_displacement", "displacement", "x")
        reaction_names = ("reaction_y", "vertical_reaction", "reaction", "force", "y")

        def select_column(names):
            return next((columns[name] for name in names if name in columns), None)

        x_col, y_col = select_column(displacement_names), select_column(reaction_names)
        if x_col is None or y_col is None:
            raise KeyError(f"{path}: displacement or reaction column missing. Columns: {reader.fieldnames}")

        x, y = [], []
        for line, row in enumerate(reader, start=2):
            x_raw, y_raw = (row.get(x_col) or "").strip(), (row.get(y_col) or "").strip()
            if not x_raw or not y_raw:
                continue
            try:
                x.append(float(x_raw))
                y.append(float(y_raw))
            except ValueError as exc:
                raise ValueError(f"{path}, line {line}: non-numeric value.") from exc

    return x, y


def load_curve(spec):
    source = spec["data"]
    label = spec["label"]
    if source == "manual":
        return prepare_curve(spec["x"], spec["y"], label)

    if source == "fem_saved":
        path, method = MAIN_RESULTS_PATH, "fem"
    elif source == "ntrpim_saved":
        path, method = MAIN_RESULTS_PATH, "ntrpim"
    elif source == "ntrpim_external":
        path, method = spec.get("path", NTRPIM2_RESULTS_PATH), "ntrpim"
    else:
        raise ValueError(f"{label}: unknown data source '{source}'.")

    path = find_result_file(path)
    if path.suffix.lower() == ".npz":
        x, y = load_npz(path, method)
    elif path.suffix.lower() == ".csv":
        x, y = load_csv(path, method)
    else:
        raise ValueError(f"{path}: only NPZ and CSV files are supported.")
    return prepare_curve(x, y, label)


def set_axis_limits(ax, loaded):
    for axis, index, limits in (("x", 1, X_LIMITS), ("y", 2, Y_LIMITS)):
        if limits is not None:
            getattr(ax, f"set_{axis}lim")(*limits)
            continue
        values = np.concatenate([item[index] for item in loaded])
        minimum, maximum = min(0.0, float(values.min())), float(values.max())
        width = max(maximum - minimum, 1e-12)
        getattr(ax, f"set_{axis}lim")(minimum - 0.02 * width, maximum + 0.05 * width)


def plot_curves(curves=CURVES, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []

    for spec in curves:
        x, y = load_curve(spec)
        loaded.append((spec, x, y))
        print(f"{spec['label']}: n={len(x)}, x=[{x.min():.6g}, {x.max():.6g}], "
              f"reaction=[{y.min():.6g}, {y.max():.6g}]")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_keys = ("color", "linestyle", "linewidth", "marker", "markersize", "zorder",
                  "markerfacecolor", "markeredgecolor", "markeredgewidth", "alpha")

    for spec, x, y in loaded:
        style = {key: spec[key] for key in style_keys if key in spec}
        style.setdefault("linewidth", 2.0)
        style.setdefault("markersize", 7.0)
        style.setdefault("zorder", 3)
        ax.plot(x, y, label=spec["label"], **style)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    set_axis_limits(ax, loaded)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", labelsize=TICK_FONTSIZE)
    ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.88,
              borderpad=0.55, labelspacing=0.45, handlelength=2.2, handletextpad=0.70, markerscale=1.0)

    png_path, pdf_path, csv_path = [output_dir / f"{OUTPUT_NAME}.{ext}" for ext in ("png", "pdf", "csv")]
    fig.tight_layout()
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["curve_label", "point_id", "prescribed_top_displacement", "vertical_reaction"])
        for spec, x, y in loaded:
            for i, (xi, yi) in enumerate(zip(x, y)):
                writer.writerow([spec["label"], i, f"{xi:.16g}", f"{yi:.16g}"])

    return png_path, pdf_path, csv_path


def main():
    for path in plot_curves():
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
