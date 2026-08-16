"""
Rebuild the standalone graphical-abstract assets.

The poster (Project/Lacuata - Beyond Point Estimates - Graphical Abstract.pdf)
was assembled in a design tool, so only some of its pieces exist as files.
This script regenerates the ones the notebook never wrote:

    figures/ga_gp_curve.png    GP posterior mean + credible bands vs margin
    figures/ga_donuts.png      blowout vs clutch scenario donuts
    figures/ga_data_badge.png  dataset scale badge

ga_heatmap_clean.png comes from the master notebook. The poster's fourth
asset, ga_problem.png, is a screenshot of a third-party win probability chart
and is deliberately not distributed with this repository.

All three are written with transparent backgrounds so they drop straight into
a poster layout.

Usage:
    python make_ga_assets.py
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
MODEL = os.path.join(HERE, "data", "processed", "best_gp_model.pkl")
SCALER = os.path.join(HERE, "data", "processed", "scaler_gp.pkl")

# Palette lifted from the poster
BLUE = "#2B5CE6"
PURPLE = "#8B5CF6"
BLOWOUT = "#4BA3E3"
CLUTCH = "#F0714F"
INK = "#1A1A2E"
MUTED = "#8A8A9A"

DPI = 300


def load_gp():
    with open(MODEL, "rb") as f:
        gp = pickle.load(f)
    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    return gp, scaler


def predict(gp, scaler, margin, seconds):
    """Posterior mean and std for one or many game states."""
    margin = np.atleast_1d(np.asarray(margin, dtype=float))
    seconds = np.full_like(margin, seconds, dtype=float)
    mean, std = gp.predict(scaler.transform(np.column_stack([margin, seconds])),
                           return_std=True)
    return np.clip(mean, 0.0, 1.0), std


# ------------------------------------------------------- ga_gp_curve.png

def make_gp_curve(gp, scaler, path):
    """Win probability vs score margin at Q4 6:00, with 90%/95% bands."""
    margins = np.arange(-25, 26, 1)
    mean, std = predict(gp, scaler, margins, 360)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for z, alpha, label in ((1.96, 0.14, "95% credible interval"),
                            (1.645, 0.26, "90% credible interval")):
        ax.fill_between(margins,
                        np.clip(mean - z * std, 0, 1) * 100,
                        np.clip(mean + z * std, 0, 1) * 100,
                        color=BLUE, alpha=alpha, linewidth=0, label=label)
    ax.plot(margins, mean * 100, color=INK, linewidth=2.6,
            label="GP posterior mean", zorder=5)

    ax.axhline(50, color="#CCCCCC", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color="#CCCCCC", linestyle="--", linewidth=1, zorder=1)

    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Score margin (home - away)", fontsize=11, color=INK)
    ax.set_ylabel("P(home win)", fontsize=11, color=INK)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Win probability with 6:00 left in the fourth",
                 fontsize=12, fontweight="bold", color=INK, pad=12)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#DDDDDD")
    ax.tick_params(colors=MUTED, labelsize=9)

    legend = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------- ga_donuts.png

def donut(ax, prob, color, label, sub, ci, epistemic, aleatoric):
    """One donut: filled arc for P(home win), hollow centre carrying the number."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.75, 1.55)

    width = 0.30
    ax.pie([prob, 1 - prob],
           colors=[color, "#E8E8EF"],
           startangle=90,
           counterclock=False,
           radius=1.0,
           wedgeprops=dict(width=width, edgecolor="none"))

    ax.text(0, 0.06, "{:.1%}".format(prob), ha="center", va="center",
            fontsize=27, fontweight="bold", color=color)
    ax.text(0, -0.26, "P(home win)", ha="center", va="center",
            fontsize=8.5, color=MUTED)

    ax.text(0, 1.36, label, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0, 1.14, sub, ha="center", va="center", fontsize=9, color=MUTED)

    ax.text(0, -1.20, "95% CI  [{:.1%}, {:.1%}]".format(*ci),
            ha="center", va="center", fontsize=9, color=INK)
    ax.text(0, -1.46,
            "epistemic {:.4f}   aleatoric {:.4f}".format(epistemic, aleatoric),
            ha="center", va="center", fontsize=8, color=MUTED,
            fontfamily="monospace")


def make_donuts(gp, scaler, path):
    """The two scenarios the paper discusses in Section IV-C."""
    scenarios = [
        # seconds remaining = (4 - quarter) * 720 + mm * 60 + ss
        ("BLOWOUT", "Home up 18, Q3 4:00  (89-71)", 18, 960, BLOWOUT),
        ("CLUTCH", "Home up 2, Q4 1:30  (105-103)", 2, 90, CLUTCH),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4))
    fig.patch.set_alpha(0.0)

    for ax, (label, sub, margin, seconds, color) in zip(axes, scenarios):
        mean, std = predict(gp, scaler, margin, seconds)
        prob, sigma = float(mean[0]), float(std[0])
        ci = (max(0.0, prob - 1.96 * sigma), min(1.0, prob + 1.96 * sigma))
        aleatoric = 0.5 - abs(prob - 0.5)
        donut(ax, prob, color, label, sub, ci, sigma, aleatoric)

    fig.subplots_adjust(wspace=0.05)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)


# ----------------------------------------------------- ga_data_badge.png

def make_data_badge(path):
    """Dataset scale, as a standalone badge."""
    stats = [
        ("6", "NBA seasons"),
        ("6,962", "games"),
        ("174,050", "observations"),
        ("634", "GP training bins"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 2.5))
    fig.patch.set_alpha(0.0)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.12, 0.15), 10.26, 2.2,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor="#F4F5FB", edgecolor="#DCDDEC", linewidth=1.4))

    slot = 10.5 / len(stats)
    for i, (value, caption) in enumerate(stats):
        centre = slot * (i + 0.5)
        ax.text(centre, 1.42, value, ha="center", va="center",
                fontsize=25, fontweight="bold", color=BLUE)
        ax.text(centre, 0.78, caption, ha="center", va="center",
                fontsize=10.5, color=MUTED)
        if i:
            ax.plot([slot * i, slot * i], [0.55, 1.95],
                    color="#DCDDEC", linewidth=1.2)

    ax.text(5.25, 0.36, "2018-19 through 2023-24  ·  sampled every 2 minutes of regulation",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")

    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)


def main():
    gp, scaler = load_gp()
    targets = [
        ("ga_gp_curve.png", lambda p: make_gp_curve(gp, scaler, p)),
        ("ga_donuts.png", lambda p: make_donuts(gp, scaler, p)),
        ("ga_data_badge.png", make_data_badge),
    ]
    for name, build in targets:
        path = os.path.join(FIGURES, name)
        build(path)
        print("wrote {} ({:,} bytes)".format(path, os.path.getsize(path)))


if __name__ == "__main__":
    main()
