
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = Path("ZZZ_Visualization_Ex1")
OUTPUT_DIR = BASE_DIR if BASE_DIR.name == RESULT_DIR.name else BASE_DIR / RESULT_DIR

OUTPUT_FIG1 = "Fig03_1_total_energy_loss.png"
OUTPUT_FIG2 = "Fig03_2_first_final_load_steps.png"

DPI = 1000
FIGSIZE_1 = (7.8, 5.6)
FIGSIZE_2 = (5.8, 8.2)

USE_ABSOLUTE_LOSS = False
USE_LOCAL_ITERATION_FOR_SMALL_PLOTS = False
SHOW_STEP_BOUNDARIES = True

LEGEND_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 11
TITLE_FONTSIZE = 18

CASES = [
    {"label": "Regular", "file": RESULT_DIR / "solution_ETRPIM_and_FEM_Regular.npz",
     "color": "blue", "linestyle": "-", "linewidth": 2.2, "zorder": 4},
    {"label": "Irregular", "file": RESULT_DIR / "solution_ETRPIM_Irregular.npz",
     "color": "red", "linestyle": "--", "linewidth": 2.2, "zorder": 5},
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})


def find_result_file(file):
    candidates = [BASE_DIR / file, Path.cwd() / file, BASE_DIR.parent / file, BASE_DIR / file.name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Result file not found: {file}\nChecked:\n" + "\n".join(map(str, candidates)))


def load_case(config):
    path = find_result_file(config["file"])
    with np.load(path, allow_pickle=True) as data:
        keys = ("trpim_history", "history", "TRPIM_NNS_history", "nns_history")
        key = next((name for name in keys if name in data.files), None)
        if key is None:
            raise KeyError(f"No full history found in {path}. Available keys: {data.files}")
        history = np.asarray(data[key], dtype=float)

    # Columns 0, 1 and 4 contain iteration, load step and energy loss.
    if history.ndim != 2 or history.shape[1] < 5:
        raise ValueError(f"Invalid history shape in {path}: {history.shape}")

    iteration, step, loss = history[:, 0], history[:, 1], history[:, 4]
    valid = np.isfinite(iteration) & np.isfinite(step) & np.isfinite(loss) & (iteration > 0) & (step > 0)
    iteration, step, loss = iteration[valid], step[valid].astype(int), loss[valid]

    if iteration.size == 0:
        raise ValueError(f"No valid history data in {path}")
    if USE_ABSOLUTE_LOSS:
        loss = np.abs(loss)
    if np.any(np.diff(iteration) < 0):
        print(f"Warning: iteration restarts in {config['label']}; using row numbers instead.")
        iteration = np.arange(1, len(iteration) + 1, dtype=float)

    return {**config, "path": path, "key": key, "iteration": iteration,
            "step": step, "loss": loss, "steps": np.unique(step)}


def plot_curve(ax, case, step=None):
    if step is None:
        x, y = case["iteration"], case["loss"]
    else:
        mask = case["step"] == step
        if not np.any(mask):
            raise ValueError(f"Load step {step} not found in {case['label']}")
        x, y = case["iteration"][mask], case["loss"][mask]
        if USE_LOCAL_ITERATION_FOR_SMALL_PLOTS:
            x = x - x[0]

    ax.plot(x, y, label=case["label"], color=case["color"], linestyle=case["linestyle"],
            linewidth=case["linewidth"], zorder=case["zorder"])
    return x, y


def set_limits(ax, curves, x_pad=0.03, y_pad=0.08):
    x = np.concatenate([curve[0] for curve in curves])
    y = np.concatenate([curve[1] for curve in curves])
    dx = max(float(np.ptp(x)), 1.0)
    dy = max(float(np.ptp(y)), 1e-12)
    ax.set_xlim(float(x.min()) - x_pad * dx, float(x.max()) + x_pad * dx)
    ax.set_ylim(float(y.min()) - y_pad * dy, float(y.max()) + y_pad * dy)


def style_axis(ax):
    ax.grid(True, alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=TICK_FONTSIZE)


def save_total_loss(cases):
    fig, ax = plt.subplots(figsize=FIGSIZE_1)
    curves = [plot_curve(ax, case) for case in cases]
    set_limits(ax, curves, x_pad=0.01, y_pad=0.05)

    if SHOW_STEP_BOUNDARIES:
        # Mark the beginning of each load step using the regular case.
        reference = cases[0]
        for step in reference["steps"][1:]:
            x_start = reference["iteration"][np.where(reference["step"] == step)[0][0]]
            ax.axvline(x_start, color="red", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    ax.set_xlabel("Total iteration")
    ax.set_ylabel("Total potential energy loss")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=1, fancybox=False,
              edgecolor="black", fontsize=LEGEND_FONTSIZE, borderpad=0.65,
              labelspacing=0.45, handlelength=2.6, handletextpad=0.75)

    fig.tight_layout()
    output = OUTPUT_DIR / OUTPUT_FIG1
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def save_first_final_steps(cases):
    common_steps = set(cases[0]["steps"])
    for case in cases[1:]:
        common_steps &= set(case["steps"])
    if not common_steps:
        raise ValueError("The cases have no common load steps to compare.")

    first_step, final_step = min(common_steps), max(common_steps)
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_2)

    for ax, step, title in zip(axes, (first_step, final_step), ("First", "Final")):
        curves = [plot_curve(ax, case, step) for case in cases]
        set_limits(ax, curves)
        ax.set_title(f"{title} load step ({step})")
        ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE - 1)
        ax.set_ylabel("Total potential energy loss", fontsize=AXIS_LABEL_FONTSIZE - 1)
        style_axis(ax)

    axes[0].legend(loc="best", frameon=True, framealpha=1, fancybox=False,
                   edgecolor="black", fontsize=LEGEND_FONTSIZE - 1, borderpad=0.55,
                   labelspacing=0.4, handlelength=2.3, handletextpad=0.65)

    fig.tight_layout()
    output = OUTPUT_DIR / OUTPUT_FIG2
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    cases = [load_case(config) for config in CASES]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"{case['label']}: {case['path']}")
        print(f"  Key: {case['key']} | Points: {len(case['iteration'])} | "
              f"Iterations: {case['iteration'].min():g}-{case['iteration'].max():g} | "
              f"Loss: {case['loss'].min():g}-{case['loss'].max():g} | "
              f"Load steps: {case['steps'].tolist()}")

    print(f"\nSaved: {save_total_loss(cases).resolve()}")
    print(f"Saved: {save_first_final_steps(cases).resolve()}")


if __name__ == "__main__":
    main()
