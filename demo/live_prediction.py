"""
Live NBA win probability prediction with GP uncertainty.

Console version of the live prediction cell from notebooks/nba_gp_complete.ipynb.
Loads the trained Gaussian Process model and feature scaler, prompts for a game
state, and prints the posterior mean, epistemic and aleatoric uncertainty,
credible intervals, and a context classification.

The model artifacts are not tracked in git (see .gitignore). Run the master
notebook once to regenerate them into data/processed/.

Usage:
    python demo/live_prediction.py
"""

import os
import pickle
import sys
import warnings

import numpy as np

# The pickled model was fitted under scikit-learn 1.6.1. Loading it under a
# different minor version raises InconsistentVersionWarning, which is noise for
# a demo; the predictions match the notebook exactly.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# ---------------------------------------------------------------- constants

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "data", "processed", "best_gp_model.pkl")
SCALER_PATH = os.path.join(REPO_ROOT, "data", "processed", "scaler_gp.pkl")

QUARTER_SECONDS = 720          # 12 minutes of regulation per quarter
OVERTIME_SECONDS = 300         # 5 minutes per overtime period
REGULATION_QUARTERS = 4

Z_90 = 1.645
Z_95 = 1.96

BLOWOUT_MARGIN = 15            # |margin| >= 15
CLUTCH_MARGIN = 5              # |margin| <= 5 and
CLUTCH_SECONDS = 300           #   <= 5 minutes remaining


# ------------------------------------------------------------------ loading

def load_artifacts():
    """Load the trained GP and its feature scaler, or exit with guidance."""
    missing = [p for p in (MODEL_PATH, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        print("Could not find the trained model artifacts:")
        for path in missing:
            print("  missing: {}".format(path))
        print("\nThese files are excluded from version control. Run")
        print("notebooks/nba_gp_complete.ipynb to regenerate them, then retry.")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ------------------------------------------------------------------- input

def ask_text(prompt, default):
    """Read a non-empty string, falling back to a default."""
    value = input(prompt).strip()
    return value if value else default


def ask_int(prompt, low=None, high=None):
    """Read an integer, re-prompting until it parses and fits the range."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("  -> Enter a whole number.")
            continue
        if low is not None and value < low:
            print("  -> Must be at least {}.".format(low))
            continue
        if high is not None and value > high:
            print("  -> Must be at most {}.".format(high))
            continue
        return value


def ask_period():
    """Read the period. 1-4 is regulation, 5+ is overtime (OT1 = 5)."""
    while True:
        raw = input("  Quarter (1-4, or 5+ for OT): ").strip().upper()
        if raw.startswith("OT"):
            suffix = raw[2:].strip()
            period = REGULATION_QUARTERS + (int(suffix) if suffix.isdigit() else 1)
            return period
        try:
            period = int(raw)
        except ValueError:
            print("  -> Enter 1-4, or 5 / OT1 for overtime.")
            continue
        if period < 1:
            print("  -> Period must be 1 or greater.")
            continue
        return period


def ask_clock(period):
    """Read time left in the period as M:SS, MM:SS, or a plain minute count."""
    period_length = QUARTER_SECONDS if period <= REGULATION_QUARTERS else OVERTIME_SECONDS
    limit_label = "12:00" if period <= REGULATION_QUARTERS else "5:00"

    while True:
        raw = input("  Time left in period (M:SS, max {}): ".format(limit_label)).strip()
        if not raw:
            print("  -> Enter a clock value such as 1:30.")
            continue

        if ":" in raw:
            parts = raw.split(":")
            if len(parts) != 2 or not all(p.strip().lstrip("-").isdigit() for p in parts):
                print("  -> Format is M:SS, for example 1:30.")
                continue
            minutes, seconds = int(parts[0]), int(parts[1])
        else:
            if not raw.lstrip("-").replace(".", "", 1).isdigit():
                print("  -> Format is M:SS, for example 1:30.")
                continue
            minutes, seconds = int(float(raw)), 0

        if minutes < 0 or seconds < 0:
            print("  -> Time cannot be negative.")
            continue
        if seconds >= 60:
            print("  -> Seconds must be under 60.")
            continue

        remaining = minutes * 60 + seconds
        if remaining > period_length:
            print("  -> A period is only {} long.".format(limit_label))
            continue
        return remaining


def seconds_remaining(period, clock_seconds):
    """Regulation seconds remaining. Overtime clamps to 0 (end of regulation)."""
    if period > REGULATION_QUARTERS:
        return 0
    return (REGULATION_QUARTERS - period) * QUARTER_SECONDS + clock_seconds


# ---------------------------------------------------------------- inference

def classify_context(margin, total_seconds):
    """Bucket the game state the same way the paper's uncertainty table does."""
    if abs(margin) >= BLOWOUT_MARGIN:
        return (
            "BLOWOUT",
            "Outcome is likely, but the model has sparse training data at "
            "extreme margins. Read the interval, not just the point estimate.",
        )
    if abs(margin) <= CLUTCH_MARGIN and total_seconds <= CLUTCH_SECONDS:
        return (
            "CLUTCH",
            "The model is confident in the estimate, but the outcome itself is "
            "close to a coin flip. High aleatoric uncertainty.",
        )
    return (
        "NORMAL",
        "Reliable estimate. Standard game state with dense training coverage.",
    )


def predict(model, scaler, margin, total_seconds):
    """Posterior mean, std, and clipped credible intervals for one game state."""
    features = scaler.transform(np.array([[margin, total_seconds]], dtype=float))
    mean, std = model.predict(features, return_std=True)
    prob = float(np.clip(mean[0], 0.0, 1.0))
    sigma = float(std[0])

    return {
        "prob": prob,
        "std": sigma,
        "aleatoric": 0.5 - abs(prob - 0.5),
        "ci90": (max(0.0, prob - Z_90 * sigma), min(1.0, prob + Z_90 * sigma)),
        "ci95": (max(0.0, prob - Z_95 * sigma), min(1.0, prob + Z_95 * sigma)),
    }


# ------------------------------------------------------------------ output

def report(home, away, home_score, away_score, period, clock, total_seconds, pred):
    margin = home_score - away_score
    context, note = classify_context(margin, total_seconds)
    line = "=" * 58

    if period <= REGULATION_QUARTERS:
        period_label = "Q{}".format(period)
    else:
        period_label = "OT{}".format(period - REGULATION_QUARTERS)

    if margin > 0:
        margin_label = "{} +{}".format(home, margin)
    elif margin < 0:
        margin_label = "{} +{}".format(away, -margin)
    else:
        margin_label = "TIED"

    print()
    print(line)
    print("  {} {} - {} {}".format(home, home_score, away_score, away))
    print("  {} {}:{:02d}  |  {}".format(
        period_label, clock // 60, clock % 60, margin_label))
    print(line)
    print("  P({} win):        {:>7.1%}".format(home, pred["prob"]))
    print("  P({} win):        {:>7.1%}".format(away, 1 - pred["prob"]))
    print()
    print("  Epistemic (GP std):  {:>7.4f}".format(pred["std"]))
    print("  Aleatoric proxy:     {:>7.4f}".format(pred["aleatoric"]))
    print()
    print("  90% credible interval: [{:.1%}, {:.1%}]".format(*pred["ci90"]))
    print("  95% credible interval: [{:.1%}, {:.1%}]".format(*pred["ci95"]))
    print()
    print("  Context: {}".format(context))
    for chunk in wrap(note, 52):
        print("    {}".format(chunk))
    if period > REGULATION_QUARTERS:
        print()
        print("  NOTE: the model was trained on regulation only. Overtime is")
        print("  scored at 0 seconds remaining, which is an extrapolation.")
    if margin == 0:
        print()
        print("  NOTE: tied game. The estimate reflects home court advantage")
        print("  and time remaining alone.")
    print(line)


def wrap(text, width):
    """Minimal greedy word wrap, keeps the script dependency-free."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# -------------------------------------------------------------------- main

def main():
    print("=" * 58)
    print("  NBA WIN PROBABILITY: LIVE GP PREDICTION")
    print("  Gaussian Process posterior with credible intervals")
    print("=" * 58)
    print()

    model, scaler = load_artifacts()

    home = ask_text("  Home team: ", "HOME").upper()
    away = ask_text("  Away team: ", "AWAY").upper()
    home_score = ask_int("  Home score: ", low=0)
    away_score = ask_int("  Away score: ", low=0)
    period = ask_period()
    clock = ask_clock(period)

    margin = home_score - away_score
    total_seconds = seconds_remaining(period, clock)

    pred = predict(model, scaler, margin, total_seconds)
    report(home, away, home_score, away_score, period, clock, total_seconds, pred)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(130)
