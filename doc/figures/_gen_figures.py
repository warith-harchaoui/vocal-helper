"""Generate vocal-helper tech-report figures from study JSON outputs.

All figures land as SVG under ``~/vocal-helper/doc/figures/`` and use
the canonical pdbms / vocal-helper figure palette
(``harchaoui.org/warith/colors/`` "Base Palette with Contrasts").

Re-run after each new study to refresh.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- canonical palette ------------------------------------------------
PRIMARY       = "#007AFF"  # Blue
ACCENT_RECO   = "#FF3B30"  # Red — chosen operating point
ACCENT_RUNNER = "#FF9500"  # Orange — runner-up
ACCENT_ALT    = "#79DBDC"  # Turquoise — third alternate
ACCENT_DEEP   = "#AF52DE"  # Purple
ACCENT_GREEN  = "#28CD41"  # Green — positive
ACCENT_PINK   = "#FF2D55"
ACCENT_YELLOW = "#FFCC00"
MUTED         = "#7F8C8D"

mpl.rcParams.update({
    "font.family": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

RUN_LOGS = Path(
    "/Volumes/orange-dev/extra/pdbms-scratch/run-logs"
)
OUT = Path(__file__).resolve().parent

# Human-readable descriptors of the AMI dev-slice meetings we use.
# Figures should NEVER show the raw AMI code alone — always a
# reader-friendly label. The code goes in parentheses at most, in
# the caption / title if needed for reproducibility.
MEETING_LABEL_SHORT = {
    "IS1008a": "Kickoff\n16 min · 4 spk",
    "IS1008b": "Kickoff #2\n29 min · 4 spk",
    "ES2011a": "Design #1\n19 min · 4 spk",
    "ES2011b": "Design #2\n26 min · 4 spk",
    "ES2011c": "Design #3\n35 min · 4 spk",
    "ES2011d": "Design #4 (long)\n33 min · 4 spk",
    "TS3004a": "Phone-style\n22 min · 4 spk",
    "TS3004b": "Phone-style #2\n37 min · 4 spk",
}


# ---------------------------------------------------------------------------
# Figure 1 — LLM 7-model Pareto on a 16-min meeting
# ---------------------------------------------------------------------------


def fig_llm_model_pareto():
    data = json.loads((RUN_LOGS / "vocal_helper_llm_model_size_2026-06-30.json").read_text())
    rows = data["rows"]
    labels = [r["model"] for r in rows]
    rtfs   = [r["rtf"]    for r in rows]
    coss   = [r["cos_sim"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Pareto front : non-dominated points (higher cos_sim or lower RTF).
    front = []
    for i, (_lab, r, c) in enumerate(zip(labels, rtfs, coss, strict=True)):
        dominated = any(
            (ro["rtf"] < r and ro["cos_sim"] >= c) or
            (ro["rtf"] <= r and ro["cos_sim"] > c)
            for j, ro in enumerate(rows) if j != i
        )
        if not dominated:
            front.append(i)

    # Background : dominated points in muted grey.
    for i, (lab, r, c) in enumerate(zip(labels, rtfs, coss, strict=True)):
        if i in front:
            continue
        ax.scatter(r, c, s=110, color=MUTED, alpha=0.55, zorder=2,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(lab, (r, c), xytext=(7, -2),
                    textcoords="offset points", fontsize=7.5, color=MUTED)

    # Pareto front : highlighted.
    fx = [rtfs[i] for i in front]
    fy = [coss[i] for i in front]
    # Connect the front from low-RTF to high-RTF.
    order = sorted(range(len(fx)), key=lambda k: fx[k])
    fx_s = [fx[k] for k in order]
    fy_s = [fy[k] for k in order]
    ax.plot(fx_s, fy_s, color=ACCENT_RECO, linewidth=1.6,
            linestyle="--", alpha=0.9, zorder=3, label="Pareto front")

    for i in front:
        is_default = (labels[i] == "gemma3:4b")
        color = ACCENT_GREEN if is_default else ACCENT_RECO
        size = 180 if is_default else 130
        ax.scatter(rtfs[i], coss[i], s=size, color=color, zorder=4,
                   edgecolor="white", linewidth=1.6,
                   label="vocal-helper default" if is_default else None)
        prefix = "★ " if is_default else ""
        ax.annotate(prefix + labels[i], (rtfs[i], coss[i]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5, fontweight="bold" if is_default else "normal",
                    color="#222")

    ax.set_xlabel("RTF (wall_time / audio_duration ; lower is better)")
    ax.set_ylabel("cos_sim vs offline single-shot reference (higher is better)")
    ax.set_title("LLM analyst — 7-model Pareto on a 16-minute team meeting\n"
                 "(4 speakers, technical kickoff ; refresh every 60 s of "
                 "content ; study : studies/llm_model_size_sweep.py, 2026-06-30)",
                 fontsize=10, loc="left")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    out_path = OUT / "fig-llm-model-pareto.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Whisper bias-prompt impact (WER drop) on AMI
# ---------------------------------------------------------------------------


def fig_whisper_prompt():
    data = json.loads((RUN_LOGS / "vocal_helper_whisper_prompt_lang_2026-06-30.json").read_text())
    per = data["per_meeting"]
    meetings = list(per.keys())
    configs = ["auto-no-prompt", "auto-bias", "en-no-prompt", "en-bias"]
    n_groups = len(meetings)
    n_bars = len(configs)
    width = 0.18

    colors = {
        "auto-no-prompt": MUTED,
        "auto-bias":      ACCENT_RECO,
        "en-no-prompt":   ACCENT_RUNNER,
        "en-bias":        ACCENT_GREEN,
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_base = list(range(n_groups))
    for j, cfg in enumerate(configs):
        wers = [per[m][cfg][0] for m in meetings]
        xs = [x + (j - (n_bars - 1) / 2) * width for x in x_base]
        ax.bar(xs, wers, width=width, color=colors[cfg],
               label=cfg, edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [MEETING_LABEL_SHORT.get(m, m) for m in meetings],
        fontsize=8,
    )
    ax.set_ylabel("Word Error Rate (lower is better)")
    ax.set_title("Whisper bias prompt drops transcription error by 15-25 percentage points\n"
                 "on two team meetings from AMI (technical kickoff · 16 min, "
                 "design meeting · 19 min ; both 4 speakers)",
                 fontsize=10, loc="left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = OUT / "fig-whisper-bias-prompt.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Diar embedding backend : separability margin
# ---------------------------------------------------------------------------


def fig_diar_embedding():
    data = json.loads((RUN_LOGS / "vocal_helper_diar_embedding_2026-06-30.json").read_text())
    pooled = data["pooled"]
    backends = ["pyannote", "titanet"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left : separability margin.
    margins = [pooled[b]["margin"] for b in backends]
    colors = [MUTED, ACCENT_GREEN]
    axes[0].bar(backends, margins, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3)
    for b, m in zip(backends, margins, strict=True):
        axes[0].annotate(f"{m:.3f}", xy=(b, m), xytext=(0, 4),
                         textcoords="offset points", ha="center",
                         fontsize=10, fontweight="bold")
    axes[0].set_title("Separability margin (inter − intra cosine)\n"
                      "higher = cleaner per-segment clustering",
                      fontsize=9.5, loc="left")
    axes[0].set_ylabel("margin")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].set_axisbelow(True)

    # Right : per-call wall.
    walls = [pooled[b]["wall_per_call_ms"] for b in backends]
    axes[1].bar(backends, walls, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3)
    for b, w in zip(backends, walls, strict=True):
        axes[1].annotate(f"{w:.0f} ms", xy=(b, w), xytext=(0, 4),
                         textcoords="offset points", ha="center",
                         fontsize=10, fontweight="bold")
    axes[1].set_title("Per-call wall time on Apple Silicon\n"
                      "lower is better — 45 ms still negligible in streaming",
                      fontsize=9.5, loc="left")
    axes[1].set_ylabel("ms / segment")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[1].set_axisbelow(True)

    fig.suptitle("Speaker embedding backend — TitaNet separates 76 % better than pyannote\n"
                 "on two team meetings (kickoff · 16 min, design meeting · 19 min ; "
                 "study : studies/diar_embedding_backend.py, 2026-06-30)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT / "fig-diar-embedding-backend.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — LLM cadence single + multi : RTF vs cos_sim
# ---------------------------------------------------------------------------


def fig_llm_cadence():
    single = json.loads((RUN_LOGS / "vocal_helper_llm_cadence_2026-06-30.json").read_text())
    multi  = json.loads((RUN_LOGS / "vocal_helper_llm_cadence_multi_2026-06-30.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharey=True)

    # --- single-meeting plot (IS1008a) ---
    single_rows = single["configs"]
    s_labels = [r["label"] for r in single_rows]
    s_rtfs   = [r["rtf"]    for r in single_rows]
    s_coss   = [r["cos_sim"] for r in single_rows]
    for lab, r, c in zip(s_labels, s_rtfs, s_coss, strict=True):
        is_winner = (lab == "t=60s")
        col = ACCENT_GREEN if is_winner else PRIMARY
        size = 130 if is_winner else 70
        axes[0].scatter(r, c, color=col, s=size, edgecolor="white",
                        linewidth=1.4, zorder=3)
        prefix = "★ " if is_winner else ""
        axes[0].annotate(prefix + lab, (r, c), xytext=(6, 5),
                         textcoords="offset points", fontsize=7.5,
                         fontweight="bold" if is_winner else "normal")
    axes[0].set_title("On one 16-minute team meeting", fontsize=9.5, loc="left")
    axes[0].set_xlabel("RTF (lower = better)")
    axes[0].set_ylabel("cos_sim vs offline reference (higher = better)")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    axes[0].set_xscale("log")

    # --- multi-meeting plot (pooled median over 4 meetings) ---
    pooled = multi["pooled"]
    m_labels = list(pooled.keys())
    m_rtfs   = [pooled[k][0] for k in m_labels]
    m_coss   = [pooled[k][2] for k in m_labels]
    for lab, r, c in zip(m_labels, m_rtfs, m_coss, strict=True):
        is_winner = (lab == "t=60s")
        col = ACCENT_GREEN if is_winner else PRIMARY
        size = 130 if is_winner else 70
        axes[1].scatter(r, c, color=col, s=size, edgecolor="white",
                        linewidth=1.4, zorder=3)
        prefix = "★ " if is_winner else ""
        axes[1].annotate(prefix + lab, (r, c), xytext=(6, 5),
                         textcoords="offset points", fontsize=7.5,
                         fontweight="bold" if is_winner else "normal")
    axes[1].set_title("Median over four team meetings (16 to 33 min each)", fontsize=9.5, loc="left")
    axes[1].set_xlabel("RTF (lower = better)")
    axes[1].grid(True, linestyle=":", alpha=0.4)
    axes[1].set_xscale("log")

    fig.suptitle("Rolling-summary cadence — refresh every 60 s of content is the best trade-off\n"
                 "between speed (RTF) and similarity to the offline reference summary "
                 "(studies : studies/llm_cadence_sweep{,_multi}.py, 2026-06-30)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT / "fig-llm-cadence.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    print("Generating vocal-helper tech-report figures …")
    fig_llm_model_pareto()
    fig_whisper_prompt()
    fig_diar_embedding()
    fig_llm_cadence()
    print("[done]")


if __name__ == "__main__":
    main()
