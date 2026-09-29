"""Traffic-replay animation of the trained CNN-LSTM IDS (demo / slides).

Uses ONLY real artefacts of the finished run:
  * outputs/models/CNN-LSTM.keras          the trained model
  * outputs/cache/X_test.npy, y_test.npy,  held-out CICIoT2023 test flows
    sub_test_codes.npy + meta.json          (+ their fine-grained attack labels)
  * outputs/metrics/history_CNN-LSTM.csv    for the training-curve GIF
Raw CSVs are never read. All predictions come from ONE batched model.predict
call before rendering; the animation only replays them.

The flow stream is RE-SAMPLED for readability (long benign stretches, then
named attack bursts) - its running metrics are a demo subset, NOT the reported
test-set metrics (those are in outputs/metrics/paper_comparison.*).

usage:
  python simulation.py                       # MP4 (1920x1080, 30 fps) + GIF + stills
  python simulation.py --preview             # 5-second preview
  python simulation.py --gif_only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import confusion_matrix

from config import Config
from preprocessing import label_family

log = logging.getLogger("simulation")

# Okabe-Ito palette (colour-blind safe) + a glyph so colour is never the only cue
OUTCOME = {"TN": ("#56B4E9", "✓ correct benign"), "TP": ("#009E73", "✓ attack caught"),
           "FP": ("#F0E442", "✗ FALSE ALARM"), "FN": ("#D55E00", "✗ MISSED ATTACK")}
METRIC_STYLE = {"Accuracy": ("#0072B2", "-"), "Precision": ("#009E73", "--"),
                "Recall": ("#E69F00", ":"), "F1": ("#CC79A7", "-.")}
BURST_C = "#D55E00"
# (family, share of the stream): benign stretches between named attack bursts
SCRIPT = [("Benign", 110), ("DDoS", 70), ("Benign", 90), ("Recon", 45), ("Benign", 80),
          ("Mirai", 60), ("Benign", 65), ("Spoofing", 40), ("Benign", 40)]
FEED_ROWS = 12


# ============================================================================
# Data: pick the replay stream and predict it in one batch
# ============================================================================
def build_stream(sub_labels: np.ndarray, n_flows: int, seed: int) -> pd.DataFrame:
    """Deterministic benign/burst script drawn without replacement from the test set."""
    rng = np.random.default_rng(seed)
    total = sum(n for _, n in SCRIPT)
    sizes = [max(5, round(n * n_flows / total)) for _, n in SCRIPT]
    sizes[0] += n_flows - sum(sizes)
    fam = np.array([label_family(s) for s in sub_labels])
    used = np.zeros(len(sub_labels), bool)
    rows = []
    for seg_no, ((family, _), size) in enumerate(zip(SCRIPT, sizes)):
        if family == "Benign":
            name, pool = "Benign", np.where((fam == "Benign") & ~used)[0]
        else:
            subs, counts = np.unique(sub_labels[(fam == family) & ~used], return_counts=True)
            ok = subs[counts >= size]
            if len(ok) == 0:
                raise ValueError(f"not enough {family} test flows for a {size}-flow burst")
            name = str(rng.choice(ok))
            pool = np.where((sub_labels == name) & ~used)[0]
        pick = rng.choice(pool, size, replace=False)
        used[pick] = True
        rows += [{"test_index": int(i), "segment": seg_no, "burst": None if family == "Benign" else name}
                 for i in pick]
    df = pd.DataFrame(rows)
    df.insert(0, "step", np.arange(1, len(df) + 1))
    return df


def predict_stream(stream: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    import keras

    d = cfg.array_cache_dir
    meta = json.loads((d / "meta.json").read_text())
    X = np.load(d / "X_test.npy", mmap_mode="r")
    y = np.load(d / "y_test.npy", mmap_mode="r")
    names = np.asarray(meta["sub_test_names"])[np.load(d / "sub_test_codes.npy")]
    idx = stream["test_index"].to_numpy()
    model = keras.models.load_model(cfg.model_dir / "CNN-LSTM.keras")
    proba = model.predict(np.asarray(X[idx], np.float32)[..., None], batch_size=1024, verbose=0)[:, 1]
    s = stream.copy()
    s["sub_label"] = names[idx]
    s["true"] = np.asarray(y[idx]).astype(int)
    assert (s["true"] == (s["sub_label"].map(label_family) != "Benign")).all(), "label cache mismatch"
    s["p_malicious"] = proba
    s["pred"] = (proba >= cfg.threshold).astype(int)
    s["outcome"] = np.select([(s.true == 1) & (s.pred == 1), (s.true == 0) & (s.pred == 0),
                              (s.true == 0) & (s.pred == 1)], ["TP", "TN", "FP"], "FN")
    for k in ("TP", "TN", "FP", "FN"):
        s[f"cum_{k}"] = (s["outcome"] == k).cumsum()
    tp, tn, fp, fn = (s[f"cum_{k}"].to_numpy(float) for k in ("TP", "TN", "FP", "FN"))
    with np.errstate(invalid="ignore", divide="ignore"):
        s["run_accuracy"] = (tp + tn) / s["step"]
        s["run_precision"] = tp / (tp + fp)
        s["run_recall"] = tp / (tp + fn)
        s["run_f1"] = 2 * tp / (2 * tp + fp + fn)
    return s


# ============================================================================
# Figure
# ============================================================================
class ReplayFigure:
    """All artists created once; update(k) only mutates them (fast, no re-layout)."""

    def __init__(self, s: pd.DataFrame, threshold: float, dpi: float):
        self.s, self.thr, self.n = s, threshold, len(s)
        plt.rcParams.update({"font.family": "DejaVu Sans"})
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi, facecolor="white")
        f = self.fig
        f.text(0.02, 0.955, "CNN-LSTM IDS, Live Traffic Replay", fontsize=28, weight="bold", va="center")
        self.t_count = f.text(0.45, 0.955, "", fontsize=18, va="center")
        self.t_burst = f.text(0.02, 0.905, "", fontsize=18, va="center", weight="bold")
        self.t_alert = f.text(0.975, 0.955, "  ALERT  ", fontsize=22, weight="bold", color="white",
                              ha="right", va="center",
                              bbox=dict(boxstyle="round,pad=0.35", fc="#C0392B", ec="none"))
        n_ben = int((s.true == 0).sum())
        f.text(0.02, 0.018,
               f"Replay of {self.n} REAL held-out CICIoT2023 test flows ({n_ben} benign, {self.n - n_ben} "
               "attack), re-sampled into benign stretches + named attack bursts for readability. "
               "Predictions: trained CNN-LSTM, computed up front. Running metrics describe this demo "
               "subset only; the reported results come from the full test set (paper comparison).",
               fontsize=11.5, color="#333333", wrap=True)
        gs = f.add_gridspec(2, 2, left=0.04, right=0.985, top=0.855, bottom=0.1, hspace=0.3,
                            wspace=0.14, height_ratios=[1.12, 1])
        self._feed(f.add_subplot(gs[0, 0]))
        right = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 1.25], hspace=0.55)
        self._gauge(f.add_subplot(right[0]), f.add_subplot(right[1]))
        self._cm(f.add_subplot(gs[1, 0]))
        self._metrics(f.add_subplot(gs[1, 1]))

    # ---- panels -------------------------------------------------------------
    def _feed(self, ax):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, FEED_ROWS + 1.3)
        ax.set_title("Flow feed (latest on top)", fontsize=17, loc="left", weight="bold")
        cols = [(0.01, "Flow #"), (0.125, "True traffic type"), (0.445, "Verdict"),
                (0.61, "P(mal.)"), (0.735, "Outcome")]
        for x, h in cols:
            ax.text(x, FEED_ROWS + 0.55, h, fontsize=13.5, weight="bold", va="center")
        self.rows = []
        for j in range(FEED_ROWS):
            y = FEED_ROWS - 1 - j + 0.5
            rect = Rectangle((0, y - 0.44), 1, 0.88, lw=0, alpha=0.9)
            ax.add_patch(rect)
            texts = [ax.text(x, y, "", fontsize=13, va="center", family="DejaVu Sans Mono" if i in (0, 3)
                             else "DejaVu Sans") for i, (x, _) in enumerate(cols)]
            self.rows.append((rect, texts))
        self.rows[0][0].set_edgecolor("black")
        self.rows[0][0].set_linewidth(2.2)

    def _gauge(self, ax, ax_sp):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("P(malicious) for the current flow", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.set_title("Model output", fontsize=17, loc="left", weight="bold")
        ax.barh(0, 1, height=0.7, color="#EEEEEE")
        self.g_bar = ax.barh(0, 0, height=0.7)[0]
        ax.axvline(self.thr, color="black", lw=2.5, ls="--")
        ax.text(self.thr + 0.01, -0.55, f"threshold {self.thr}", fontsize=12, va="bottom")
        self.g_text = ax.text(1.0, 1.06, "", ha="right", va="bottom", fontsize=15, weight="bold",
                              transform=ax.transAxes)
        ax_sp.set_xlim(-60, 0)
        ax_sp.set_ylim(-0.05, 1.05)
        ax_sp.axhline(self.thr, color="black", lw=1.2, ls="--")
        ax_sp.set_ylabel("P(mal.)", fontsize=12)
        ax_sp.set_xlabel("flows ago", fontsize=12)
        ax_sp.tick_params(labelsize=11)
        ax_sp.set_title("Last 60 flows", fontsize=14, loc="left")
        (self.sp_line,) = ax_sp.plot([], [], color="#444444", lw=1.2)
        self.sp_pts = ax_sp.scatter([], [], s=34, zorder=3, edgecolors="black", linewidths=0.5)

    def _cm(self, ax):
        self.cm_ax = ax
        self.cm_img = ax.imshow(np.ones((2, 2)), cmap="Blues", norm=LogNorm(1, max(self.n, 2)))
        ax.set_xticks([0, 1], ["Benign", "Malicious"], fontsize=13)
        ax.set_yticks([0, 1], ["Benign", "Malicious"], fontsize=13)
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True", fontsize=14)
        ax.set_title("Running confusion matrix", fontsize=17, loc="left", weight="bold")
        tags = [["TN", "FP"], ["FN", "TP"]]
        self.cm_txt = [[ax.text(j, i, "", ha="center", va="center", fontsize=19, weight="bold")
                        for j in range(2)] for i in range(2)]
        self.cm_tags = tags

    def _metrics(self, ax):
        s = self.s
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("Flows processed", fontsize=14)
        ax.set_ylabel("Running score (demo subset)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3)
        ax.set_title("Running metrics", fontsize=17, loc="left", weight="bold")
        self.bursts = []
        for seg, g in s[s.burst.notna()].groupby("segment"):
            start, end = g.step.min() - 1, g.step.max()
            r = Rectangle((start, 0), 0, 1.04, color=BURST_C, alpha=0.13, lw=0)
            ax.add_patch(r)
            lbl = ax.text(start + 1.5, 0.2, g.burst.iloc[0], fontsize=10.5, color="#8B2E00",
                          rotation=90, va="bottom", ha="left", visible=False,
                          bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
            self.bursts.append((r, lbl, start, end))
        self.m_lines = {}
        for name, (c, ls) in METRIC_STYLE.items():
            (self.m_lines[name],) = ax.plot([], [], color=c, ls=ls, lw=2.6, label=name)
        handles = list(self.m_lines.values()) + [Patch(color=BURST_C, alpha=0.25, label="attack burst")]
        ax.legend(handles=handles, loc="lower right", fontsize=12, ncol=5, framealpha=0.95)

    # ---- per-frame update ----------------------------------------------------
    def update(self, k: int, blink_on: bool = True):
        s, n = self.s, self.n
        cur = s.iloc[k - 1]
        self.t_count.set_text(f"Flows processed: {k:>4d} / {n}")
        if isinstance(cur.burst, str):
            self.t_burst.set_text(f"Attack burst: {cur.burst}")
            self.t_burst.set_color(BURST_C)
        else:
            self.t_burst.set_text("Benign traffic")
            self.t_burst.set_color("#1F6F8B")
        self.t_alert.set_visible(bool(cur.pred == 1 and blink_on))

        recent = s.iloc[max(0, k - FEED_ROWS):k].iloc[::-1]
        for j, (rect, texts) in enumerate(self.rows):
            if j < len(recent):
                r = recent.iloc[j]
                col, word = OUTCOME[r.outcome]
                rect.set_facecolor(col)
                rect.set_alpha(0.95 if r.outcome in ("FP", "FN") else 0.45)
                vals = [f"{int(r.step):04d}", r.sub_label, "MALICIOUS" if r.pred else "benign",
                        f"{r.p_malicious:6.3f}", word]
                for t, v in zip(texts, vals):
                    t.set_text(v)
                    t.set_weight("bold" if r.outcome in ("FP", "FN") else "normal")
                    t.set_color("white" if r.outcome == "FN" else "black")
            else:
                rect.set_alpha(0)
                for t in texts:
                    t.set_text("")

        p = float(cur.p_malicious)
        self.g_bar.set_width(p)
        self.g_bar.set_color(OUTCOME["TP" if p >= self.thr else "TN"][0])
        self.g_text.set_text(f"P = {p:.3f}  →  {'MALICIOUS' if p >= self.thr else 'benign'}"
                             f"   (true: {'attack' if cur.true else 'benign'})")
        w = s.iloc[max(0, k - 60):k]
        x = w.step.to_numpy() - k
        self.sp_line.set_data(x, w.p_malicious)
        self.sp_pts.set_offsets(np.c_[x, w.p_malicious])
        self.sp_pts.set_color([OUTCOME[o][0] for o in w.outcome])

        cm = np.array([[cur.cum_TN, cur.cum_FP], [cur.cum_FN, cur.cum_TP]], float)
        self.cm_img.set_data(np.maximum(cm, 1))
        for i in range(2):
            for j in range(2):
                t = self.cm_txt[i][j]
                t.set_text(f"{self.cm_tags[i][j]}\n{int(cm[i, j]):,}")
                t.set_color("white" if cm[i, j] > n / 8 else "black")

        xs = s.step.iloc[:k].to_numpy()
        for name in METRIC_STYLE:
            self.m_lines[name].set_data(xs, s[f"run_{name.lower()}"].iloc[:k])
        for rect, lbl, start, end in self.bursts:
            width = max(0, min(k, end) - start)
            rect.set_width(width)
            lbl.set_visible(width > 0)
        return self.cm_matrix(k)

    def cm_matrix(self, k: int) -> np.ndarray:
        r = self.s.iloc[k - 1]
        return np.array([[r.cum_TN, r.cum_FP], [r.cum_FN, r.cum_TP]], int)


# ============================================================================
# Writers + checks
# ============================================================================
def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def probe(path: Path, exe: str) -> dict:
    """Duration / resolution / codec via ffprobe, or `ffmpeg -i` if ffprobe is absent."""
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2)}
    fp = shutil.which("ffprobe")
    if fp:
        out = json.loads(subprocess.run([fp, "-v", "error", "-print_format", "json", "-show_streams",
                                         "-show_format", str(path)], capture_output=True, text=True).stdout)
        v = next(st for st in out["streams"] if st["codec_type"] == "video")
        info.update(codec=v["codec_name"], width=v["width"], height=v["height"],
                    fps=v["r_frame_rate"], duration_s=float(out["format"]["duration"]))
        return info
    err = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    v = re.search(r"Video: (\w+).*?, (\d+)x(\d+).*?, ([\d.]+) fps", err)
    if m:
        info["duration_s"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    if v:
        info.update(codec=v[1], width=int(v[2]), height=int(v[3]), fps=float(v[4]))
    return info


def frame_plan(n: int, seconds: float, fps: float, hold_s: float = 2.0) -> list[tuple[int, int]]:
    """[(flows_processed, frame_no)]: evenly spread flows, then a hold on the final state."""
    main = max(1, int(round((seconds - hold_s) * fps)))
    plan = [(min(n, max(1, int(np.ceil((f + 1) * n / main)))), f) for f in range(main)]
    plan += [(n, main + h) for h in range(int(hold_s * fps))]
    return plan


def render(s: pd.DataFrame, cfg: Config, path: Path, fps: float, seconds: float, dpi: float,
           writer_kind: str, exe: str | None, limit_frames: int | None = None) -> Path:
    fig_obj = ReplayFigure(s, cfg.threshold, dpi)
    plan = frame_plan(len(s), seconds, fps)
    if limit_frames:
        plan = plan[:limit_frames]
    blink = max(1, int(fps / 4))  # ~2 blinks per second

    def step(item):
        k, f = item
        fig_obj.update(k, blink_on=(f // blink) % 2 == 0)
        return []

    anim = animation.FuncAnimation(fig_obj.fig, step, frames=plan, blit=False, cache_frame_data=False)
    if writer_kind == "mp4":
        plt.rcParams["animation.ffmpeg_path"] = exe
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                                        extra_args=["-pix_fmt", "yuv420p", "-crf", "21", "-preset", "medium"])
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig_obj.fig)
    log.info("wrote %s (%d frames)", path, len(plan))
    return path


def save_still(s: pd.DataFrame, cfg: Config, path: Path, k: int | None = None) -> np.ndarray:
    fig_obj = ReplayFigure(s, cfg.threshold, dpi=100)
    cm = fig_obj.update(k or len(s), blink_on=True)
    fig_obj.fig.savefig(path, dpi=100)
    plt.close(fig_obj.fig)
    return cm


def training_gif(cfg: Config, path: Path) -> Path:
    """Train/val accuracy + loss drawn epoch by epoch, best epoch marked, then a hold."""
    h = pd.read_csv(cfg.metrics_dir / "history_CNN-LSTM.csv")
    ep = np.arange(1, len(h) + 1)
    best = int(h["val_loss"].idxmin()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=90)
    lines = {}
    for ax, m in zip(axes, ("accuracy", "loss")):
        lo = min(h[m].min(), h[f"val_{m}"].min())
        hi = max(h[m].max(), h[f"val_{m}"].max())
        pad = (hi - lo) * 0.08 or 0.01
        ax.set_xlim(0.5, len(h) + 0.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(m.capitalize())
        ax.grid(alpha=0.3)
        ax.set_title(f"Training vs validation {m}")
        (lines[m],) = ax.plot([], [], "-o", color="#0072B2", ms=4, label=f"train {m}")
        (lines["val_" + m],) = ax.plot([], [], "-s", color="#E69F00", ms=4, label=f"val {m}")
        ax.legend(loc="lower right" if m == "accuracy" else "upper right")
    marks = [ax.axvline(best, color="#D55E00", ls="--", lw=1.5, visible=False) for ax in axes]
    note = fig.text(0.5, 0.01, "", ha="center", fontsize=11, color="#8B2E00")
    fig.suptitle("CNN-LSTM training on CICIoT2023 (real history)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    hold = 8
    frames = list(range(1, len(h) + 1)) + [len(h)] * hold

    def step(e):
        for key, ln in lines.items():
            ln.set_data(ep[:e], h[key].iloc[:e])
        done = e == len(h)
        for mk in marks:
            mk.set_visible(done)
        note.set_text(f"best epoch {best}: val_loss {h.val_loss.min():.4f}, "
                      f"val_accuracy {h.val_accuracy.iloc[best - 1]:.4f}" if done else "")
        return []

    anim = animation.FuncAnimation(fig, step, frames=frames, blit=False)
    anim.save(str(path), writer=animation.PillowWriter(fps=3))
    plt.close(fig)
    return path


# ============================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_flows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--seconds", type=float, default=40, help="MP4 length (30-45 s recommended)")
    ap.add_argument("--gif_fps", type=float, default=12)
    ap.add_argument("--gif_seconds", type=float, default=24)
    ap.add_argument("--gif_width", type=int, default=800)
    ap.add_argument("--output_root", type=Path, default=Path("outputs"))
    ap.add_argument("--out_dir", type=Path, default=None, help="default <output_root>/animations")
    ap.add_argument("--gif_only", action="store_true")
    ap.add_argument("--preview", action="store_true", help="render only a 5-second preview")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config(output_dir=args.output_root)
    out = args.out_dir or cfg.anim_dir
    out.mkdir(parents=True, exist_ok=True)

    stream = build_stream(np.asarray(json.loads((cfg.array_cache_dir / "meta.json").read_text())
                                     ["sub_test_names"])[np.load(cfg.array_cache_dir / "sub_test_codes.npy")],
                          args.n_flows, args.seed)
    s = predict_stream(stream, cfg)

    # the replay must be exactly the model's output: compare with sklearn
    sk = confusion_matrix(s.true, s.pred, labels=[0, 1])
    anim_cm = ReplayFigure(s, cfg.threshold, dpi=20).update(len(s))
    plt.close("all")
    assert np.array_equal(anim_cm, sk), f"animation CM {anim_cm.tolist()} != sklearn {sk.tolist()}"
    log.info("final confusion matrix matches sklearn: %s", sk.tolist())

    s.drop(columns=[c for c in s.columns if c.startswith("cum_")]).to_csv(out / "replay_log.csv", index=False)
    last = s.iloc[-1]
    demo = {"NOTE": "DEMO SUBSET (re-sampled replay stream), NOT the reported test-set metrics",
            "n_flows": len(s), "seed": args.seed, "benign_flows": int((s.true == 0).sum()),
            "attack_flows": int((s.true == 1).sum()),
            "bursts": s[s.burst.notna()].groupby("segment").burst.agg(["first", "size"])
                      .rename(columns={"first": "sub_label", "size": "flows"}).to_dict("records"),
            "tn": int(sk[0, 0]), "fp": int(sk[0, 1]), "fn": int(sk[1, 0]), "tp": int(sk[1, 1]),
            **{k: round(float(last[f"run_{k}"]), 4) for k in ("accuracy", "precision", "recall", "f1")}}
    (out / "replay_metrics.json").write_text(json.dumps(demo, indent=2))
    log.info("Demo-subset running metrics (NOT the reported test metrics): %s",
             {k: demo[k] for k in ("accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn")})

    exe = ffmpeg_exe()
    if exe is None and not args.gif_only:
        log.warning("ffmpeg not found (install: sudo apt install ffmpeg, or pip install imageio-ffmpeg)"
                    " -> GIF only")
    report = {}
    if args.preview:
        kind = "mp4" if (exe and not args.gif_only) else "gif"
        p = render(s, cfg, out / f"traffic_replay_preview.{kind}", args.fps, args.seconds, 100 if kind == "mp4"
                   else args.gif_width / 19.2, kind, exe, limit_frames=int(5 * args.fps))
        report["preview"] = probe(p, exe) if kind == "mp4" else {"file": p.name}
    else:
        if exe and not args.gif_only:
            p = render(s, cfg, out / "traffic_replay.mp4", args.fps, args.seconds, 100, "mp4", exe)
            report["mp4"] = probe(p, exe)
        gif = render(s, cfg, out / "traffic_replay.gif", args.gif_fps, args.gif_seconds,
                     args.gif_width / 19.2, "gif", exe)
        report["gif"] = {"file": gif.name, "size_mb": round(gif.stat().st_size / 1e6, 2)}
        save_still(s, cfg, out / "traffic_replay_final_16x9.png")
        for k, name in ((int(s[s.burst.notna()].step.iloc[0]) + 20, "frame_ddos_burst"),
                        (int(len(s) * 0.55), "frame_mid")):
            save_still(s, cfg, out / f"{name}.png", k)
        tg = training_gif(cfg, out / "training_curves_animated.gif")
        report["training_gif"] = {"file": tg.name, "size_mb": round(tg.stat().st_size / 1e6, 2)}
    (out / "render_report.json").write_text(json.dumps(report, indent=2))
    log.info("render report: %s", json.dumps(report))


if __name__ == "__main__":
    main()
