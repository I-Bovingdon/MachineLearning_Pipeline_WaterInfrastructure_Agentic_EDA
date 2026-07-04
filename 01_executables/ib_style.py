"""
ib_style.py  –  Visual identity module for pump_it_up_improved.ipynb
Brand: Ismael Bovingdon Castillejo · Data Engineer Portfolio
Palette & design language mirroring https://i-bovingdon.github.io/DataEngineer.com/

Usage:
    from ib_style import apply_style, COLORS, html_report, html_ai_assistant
    apply_style()   # call once at notebook top – sets all matplotlib/seaborn defaults
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ─────────────────────────────────────────────
#  BRAND TOKENS
# ─────────────────────────────────────────────
COLORS = {
    # Backgrounds
    "bg_page":        "#080e1c",
    "bg_card":        "#0d1628",
    "bg_card_alt":    "#111827",
    "bg_card_inner":  "#0a1020",
    "bg_surface":     "#131e30",

    # Accents
    "orange":         "#f97316",
    "orange_dim":     "#c2510a",
    "orange_glow":    "rgba(249,115,22,0.18)",
    "blue":           "#3b82f6",
    "blue_dim":       "#1e40af",
    "blue_glow":      "rgba(59,130,246,0.18)",
    "teal":           "#06b6d4",
    "purple":         "#8b5cf6",

    # Neutrals
    "text_primary":   "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted":     "#475569",
    "border":         "rgba(255,255,255,0.07)",
    "border_bright":  "rgba(255,255,255,0.14)",

    # Status / categorical (pump classes)
    "functional":          "#3b82f6",   # blue
    "needs_repair":        "#f97316",   # orange
    "non_functional":      "#ef4444",   # red
}

# Explicit hex for matplotlib (no rgba strings)
MPL = {
    "bg":         "#080e1c",
    "bg_card":    "#0d1628",
    "bg_ax":      "#0a1020",
    "grid":       "#1e2d45",
    "border":     "#1e2d45",
    "text":       "#e2e8f0",
    "text2":      "#94a3b8",
    "orange":     "#f97316",
    "blue":       "#3b82f6",
    "teal":       "#06b6d4",
    "purple":     "#8b5cf6",
    "red":        "#ef4444",
    "green":      "#22c55e",
    # Pump status canonical colours
    "functional":      "#3b82f6",
    "needs_repair":    "#f97316",
    "non_functional":  "#ef4444",
}

# Ordered colour cycle – blue → orange → red → teal → purple → green
CYCLE = [MPL["blue"], MPL["orange"], MPL["red"],
         MPL["teal"], MPL["purple"], MPL["green"]]

# ─────────────────────────────────────────────
#  CUSTOM COLORMAPS
# ─────────────────────────────────────────────
# Dark-navy → electric-blue  (great for heatmaps / confusion matrices)
cmap_blue = LinearSegmentedColormap.from_list(
    "ib_blue", ["#080e1c", "#0d1628", "#1e3a5f", "#3b82f6", "#93c5fd"])

# Dark → orange (intensity / importance)
cmap_orange = LinearSegmentedColormap.from_list(
    "ib_orange", ["#080e1c", "#1c0a02", "#7c2d12", "#f97316", "#fed7aa"])

# Diverging: blue ←→ orange (correlation / threshold)
cmap_div = LinearSegmentedColormap.from_list(
    "ib_div", ["#3b82f6", "#1e3a5f", "#080e1c", "#7c2d12", "#f97316"])

# Traffic-light sequential: red → orange → blue (status maps)
cmap_status = LinearSegmentedColormap.from_list(
    "ib_status", [MPL["non_functional"], MPL["needs_repair"], MPL["functional"]])

def register_cmaps():
    for cm in [cmap_blue, cmap_orange, cmap_div, cmap_status]:
        try:
            mpl.colormaps.register(cm)
            mpl.colormaps.register(cm.reversed(), name=cm.name + "_r")
        except ValueError:
            pass   # already registered

# ─────────────────────────────────────────────
#  MATPLOTLIB RCPARAMS
# ─────────────────────────────────────────────
RC = {
    # Figure
    "figure.facecolor":         MPL["bg"],
    "figure.edgecolor":         MPL["bg"],
    "figure.dpi":               130,
    "figure.titlesize":         15,
    "figure.titleweight":       "bold",

    # Axes
    "axes.facecolor":           MPL["bg_ax"],
    "axes.edgecolor":           MPL["grid"],
    "axes.labelcolor":          MPL["text2"],
    "axes.titlecolor":          MPL["text"],
    "axes.titlesize":           12,
    "axes.titleweight":         "bold",
    "axes.titlepad":            12,
    "axes.labelsize":           10,
    "axes.labelpad":            8,
    "axes.spines.top":          False,
    "axes.spines.right":        False,
    "axes.spines.left":         True,
    "axes.spines.bottom":       True,
    "axes.grid":                True,
    "axes.grid.axis":           "both",
    "axes.prop_cycle":          mpl.cycler(color=CYCLE),

    # Grid
    "grid.color":               MPL["grid"],
    "grid.linewidth":           0.5,
    "grid.alpha":               0.6,

    # Ticks
    "xtick.color":              MPL["text2"],
    "ytick.color":              MPL["text2"],
    "xtick.labelsize":          9,
    "ytick.labelsize":          9,
    "xtick.direction":          "out",
    "ytick.direction":          "out",
    "xtick.major.size":         4,
    "ytick.major.size":         4,

    # Lines & patches
    "lines.linewidth":          2.0,
    "lines.markersize":         6,
    "patch.edgecolor":          MPL["bg_ax"],
    "patch.linewidth":          0.5,

    # Legend
    "legend.facecolor":         MPL["bg_card"],
    "legend.edgecolor":         MPL["grid"],
    "legend.labelcolor":        MPL["text"],
    "legend.fontsize":          9,
    "legend.title_fontsize":    10,
    "legend.framealpha":        0.9,

    # Colorbar / image
    "image.cmap":               "ib_blue",

    # Font
    "font.family":              "sans-serif",
    "font.sans-serif":          ["Inter", "Segoe UI", "Helvetica Neue",
                                 "Arial", "DejaVu Sans"],
    "text.color":               MPL["text"],

    # Saving
    "savefig.facecolor":        MPL["bg"],
    "savefig.edgecolor":        MPL["bg"],
    "savefig.bbox":             "tight",
    "savefig.dpi":              150,
    "savefig.transparent":      False,
}

def apply_style():
    """Apply the IB brand style to all subsequent matplotlib figures."""
    register_cmaps()
    mpl.rcParams.update(RC)
    print("✓ IB brand style applied – matplotlib defaults updated.")


# ─────────────────────────────────────────────
#  FIGURE HELPERS
# ─────────────────────────────────────────────
def styled_fig(nrows=1, ncols=1, figsize=None, title=None, subtitle=None):
    """
    Create a pre-styled figure.  Returns (fig, axes).
    axes is a single Axes if nrows==ncols==1, else an ndarray.
    """
    if figsize is None:
        w = max(7 * ncols, 10)
        h = max(5 * nrows, 5)
        figsize = (w, h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=MPL["bg"])
    fig.patch.set_facecolor(MPL["bg"])

    if title:
        y = 1.02 if subtitle else 1.0
        fig.suptitle(title, fontsize=15, fontweight="bold",
                     color=MPL["text"], y=y)
    if subtitle:
        fig.text(0.5, 0.985, subtitle, ha="center", va="top",
                 fontsize=10, color=MPL["text2"])

    # Style each axes
    ax_list = np.array(axes).flatten() if hasattr(axes, "__len__") else [axes]
    for ax in ax_list:
        _style_ax(ax)

    if nrows == 1 and ncols == 1:
        return fig, axes
    return fig, axes


def _style_ax(ax):
    ax.set_facecolor(MPL["bg_ax"])
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL["grid"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MPL["text2"], labelsize=9)
    ax.xaxis.label.set_color(MPL["text2"])
    ax.yaxis.label.set_color(MPL["text2"])
    ax.title.set_color(MPL["text"])


def add_orange_accent(ax, side="left", width=3):
    """Add an orange left-border accent line to an axes (decorative)."""
    bbox = ax.get_position()
    line_x = bbox.x0 - 0.005 if side == "left" else bbox.x1 + 0.005
    ax.figure.add_artist(
        mpl.lines.Line2D(
            [line_x, line_x],
            [bbox.y0, bbox.y1],
            transform=ax.figure.transFigure,
            color=MPL["orange"],
            linewidth=width,
            solid_capstyle="round",
            clip_on=False,
        )
    )


def brand_palette_for(n, mode="blue_orange"):
    """
    Return a list of n hex colours from the brand palette.
    mode: 'blue_orange' | 'sequential_blue' | 'sequential_orange' | 'status'
    """
    if mode == "status":
        base = [MPL["functional"], MPL["needs_repair"], MPL["non_functional"]]
        return (base * ((n // 3) + 1))[:n]
    if mode == "sequential_blue":
        return [mcolors.to_hex(cmap_blue(i / max(n - 1, 1))) for i in range(n)]
    if mode == "sequential_orange":
        return [mcolors.to_hex(cmap_orange(i / max(n - 1, 1))) for i in range(n)]
    # default: alternate blue/orange/teal/red/purple
    return (CYCLE * ((n // len(CYCLE)) + 1))[:n]


# ─────────────────────────────────────────────
#  CONFUSION MATRIX HELPER
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm_array, labels, title="Confusion Matrix",
                          acc=None, ax=None, cmap=None):
    """
    Plot a styled confusion matrix.
    cm_array: 2-D numpy array
    labels: list of class names
    """
    import matplotlib.ticker as ticker

    if ax is None:
        fig, ax = styled_fig(figsize=(6, 5), title=title)
    else:
        fig = ax.figure

    cmap = cmap or cmap_blue
    n = cm_array.shape[0]
    im = ax.imshow(cm_array, cmap=cmap, aspect="auto",
                   vmin=0, vmax=cm_array.max())

    # Cell labels
    thresh = cm_array.max() / 2
    for i in range(n):
        for j in range(n):
            val = cm_array[i, j]
            colour = MPL["text"] if val < thresh else MPL["bg"]
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    color=colour, fontsize=10, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       color=MPL["text2"], fontsize=9)
    ax.set_yticklabels(labels, color=MPL["text2"], fontsize=9)
    ax.set_xlabel("Predicted label", color=MPL["text2"])
    ax.set_ylabel("True label", color=MPL["text2"])

    full_title = title
    if acc is not None:
        full_title += f"  (acc={acc:.3f})"
    ax.set_title(full_title, color=MPL["text"], fontsize=11, fontweight="bold", pad=12)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(color=MPL["text2"])
    return fig, ax


# ─────────────────────────────────────────────
#  GEOGRAPHIC MAP HELPER
# ─────────────────────────────────────────────
STATUS_SCATTER_STYLE = {
    "functional":           dict(color=MPL["functional"],      alpha=0.35, s=4, zorder=2),
    "functional needs repair": dict(color=MPL["needs_repair"], alpha=0.55, s=5, zorder=3),
    "non functional":       dict(color=MPL["non_functional"],  alpha=0.45, s=4, zorder=2),
}

def style_geo_ax(ax, title="Geographic Distribution by Status"):
    """Apply brand styling to a geo scatter axes."""
    _style_ax(ax)
    ax.set_facecolor("#070c18")   # extra-dark ocean floor feel
    ax.set_title(title, color=MPL["text"], fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", color=MPL["text2"])
    ax.set_ylabel("Latitude", color=MPL["text2"])
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL["orange"] + "66")   # faint orange border


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE COLOUR HELPER
# ─────────────────────────────────────────────
def importance_colors(values, cmap=None):
    """
    Map an array of importance values to hex colours using ib_orange cmap.
    Returns list of hex strings same length as values.
    """
    cmap = cmap or cmap_orange
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    return [mcolors.to_hex(cmap(norm(v))) for v in values]


# ─────────────────────────────────────────────
#  HTML REPORT TEMPLATE
# ─────────────────────────────────────────────
_HTML_HEAD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#080e1c;--bg-card:#0d1628;--bg-inner:#0a1020;
    --orange:#f97316;--orange-dim:#c2510a;
    --blue:#3b82f6;--blue-dim:#1e40af;
    --teal:#06b6d4;--purple:#8b5cf6;
    --text:#e2e8f0;--text2:#94a3b8;--text3:#475569;
    --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.14);
    --r:12px;
  }}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;
        font-size:14px;line-height:1.6;padding:32px 24px;}}

  /* Header */
  .report-header{{
    background:linear-gradient(135deg,#0d1628 0%,#111827 60%,#1c0a02 100%);
    border:1px solid var(--border2);border-radius:var(--r);
    padding:32px 36px;margin-bottom:32px;position:relative;overflow:hidden;
  }}
  .report-header::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--orange),var(--blue));
  }}
  .report-header .eyebrow{{
    font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
    color:var(--orange);margin-bottom:8px;
  }}
  .report-header h1{{font-size:28px;font-weight:700;color:var(--text);line-height:1.2;}}
  .report-header .subtitle{{color:var(--text2);font-size:13px;margin-top:8px;}}
  .brand{{position:absolute;top:28px;right:36px;
          display:flex;align-items:center;gap:10px;}}
  .brand-badge{{background:var(--orange);color:#fff;
                width:36px;height:36px;border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:13px;letter-spacing:.04em;}}
  .brand-name{{font-size:11px;color:var(--text2);font-weight:500;}}

  /* Metrics row */
  .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
             gap:16px;margin-bottom:28px;}}
  .metric-card{{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--r);padding:20px 18px;position:relative;overflow:hidden;
  }}
  .metric-card::before{{
    content:'';position:absolute;top:0;left:0;width:3px;height:100%;
    background:var(--orange);
  }}
  .metric-card.blue::before{{background:var(--blue);}}
  .metric-card.teal::before{{background:var(--teal);}}
  .metric-card.purple::before{{background:var(--purple);}}
  .metric-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.1em;color:var(--text2);margin-bottom:6px;}}
  .metric-value{{font-size:26px;font-weight:700;color:var(--text);line-height:1;}}
  .metric-sub{{font-size:11px;color:var(--text2);margin-top:4px;}}

  /* Section */
  .section{{margin-bottom:32px;}}
  .section-title{{
    font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
    color:var(--orange);margin-bottom:16px;padding-bottom:8px;
    border-bottom:1px solid var(--border2);
  }}

  /* Card */
  .card{{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--r);padding:24px;margin-bottom:16px;position:relative;
  }}
  .card h3{{font-size:14px;font-weight:600;color:var(--text);margin-bottom:8px;}}
  .card p{{color:var(--text2);font-size:13px;}}

  /* Tag pills */
  .tags{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}}
  .tag{{background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.3);
         color:var(--blue);font-size:11px;font-weight:500;
         padding:3px 10px;border-radius:20px;}}
  .tag.orange{{background:rgba(249,115,22,0.12);border-color:rgba(249,115,22,0.3);
                color:var(--orange);}}

  /* Figures */
  .fig-wrap{{
    background:var(--bg-inner);border:1px solid var(--border);
    border-radius:var(--r);padding:16px;margin-bottom:16px;text-align:center;
  }}
  .fig-wrap img{{max-width:100%;border-radius:8px;}}
  .fig-caption{{font-size:11px;color:var(--text2);margin-top:10px;
                 font-style:italic;}}

  /* Table */
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{background:rgba(59,130,246,0.1);color:var(--blue);font-weight:600;
      font-size:11px;letter-spacing:.06em;text-transform:uppercase;
      padding:10px 12px;text-align:left;border-bottom:1px solid var(--border2);}}
  td{{padding:9px 12px;border-bottom:1px solid var(--border);color:var(--text2);}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:rgba(255,255,255,0.03);color:var(--text);}}

  /* Progress bar */
  .bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:6px;}}
  .bar-label{{width:130px;font-size:12px;color:var(--text2);text-align:right;
               flex-shrink:0;}}
  .bar-track{{flex:1;height:8px;background:var(--bg-inner);
               border-radius:4px;overflow:hidden;}}
  .bar-fill{{height:100%;border-radius:4px;transition:width .6s ease;
              background:linear-gradient(90deg,var(--blue),var(--teal));}}
  .bar-fill.orange{{background:linear-gradient(90deg,var(--orange-dim),var(--orange));}}
  .bar-val{{width:50px;font-size:12px;font-weight:600;color:var(--text);}}

  /* Footer */
  .report-footer{{
    text-align:center;padding:20px;color:var(--text3);font-size:11px;
    border-top:1px solid var(--border);margin-top:32px;
  }}
  .report-footer a{{color:var(--orange);text-decoration:none;}}
</style>
</head>
<body>
"""

_HTML_FOOT = """
<div class="report-footer">
  Built by <a href="https://i-bovingdon.github.io/DataEngineer.com/index-en.html">
  Ismael Bovingdon Castillejo</a> · Data Engineering &amp; ML ·
  <a href="https://github.com/I-Bovingdon">GitHub</a> ·
  <a href="https://www.linkedin.com/in/ismael-bovingdon-castillejo-6a775074/">LinkedIn</a>
</div>
</body></html>
"""

def html_report(title, eyebrow, subtitle, body_html, metrics=None):
    """
    Wrap body_html in the IB branded report template.
    metrics: list of dicts with keys: label, value, sub, accent ('orange'|'blue'|'teal'|'purple')
    Returns HTML string.
    """
    metrics_html = ""
    if metrics:
        cards = ""
        for m in metrics:
            accent = m.get("accent", "")
            accent_cls = f" {accent}" if accent in ("blue", "teal", "purple") else ""
            cards += (
                f'<div class="metric-card{accent_cls}">'
                f'<div class="metric-label">{m["label"]}</div>'
                f'<div class="metric-value">{m["value"]}</div>'
                f'<div class="metric-sub">{m.get("sub","")}</div>'
                f'</div>'
            )
        metrics_html = f'<div class="metrics">{cards}</div>'

    header = (
        '<div class="report-header">'
        f'<div class="brand"><div class="brand-badge">IB</div>'
        f'<div class="brand-name">Data Engineering</div></div>'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1>'
        f'<div class="subtitle">{subtitle}</div>'
        '</div>'
    )

    return _HTML_HEAD.format(title=title) + header + metrics_html + body_html + _HTML_FOOT


# ─────────────────────────────────────────────
#  GRADIO / AI ASSISTANT HTML TEMPLATE
# ─────────────────────────────────────────────
def html_ai_assistant(project_title, project_subtitle, context_blurb,
                       suggested_questions=None, highlight_numbers=None):
    """
    Render the branded AI assistant interface HTML string (for use with gr.HTML).
    highlight_numbers: list of (number_str, label) tuples, e.g. [("59,400","water pumps")]
    """
    nums_html = ""
    if highlight_numbers:
        for num, lbl in highlight_numbers:
            nums_html += (
                f' <strong style="color:var(--orange);font-weight:700;">'
                f'{num}</strong> {lbl}'
            )

    blurb_html = context_blurb
    for num, lbl in (highlight_numbers or []):
        blurb_html = blurb_html.replace(
            f"{num} {lbl}",
            f'<span style="color:var(--orange);font-weight:700;">{num} {lbl}</span>'
        )

    sq_html = ""
    if suggested_questions:
        pills = "".join(
            f'<button onclick="document.getElementById(\'ib-input\').value=this.innerText" '
            f'style="background:var(--bg-inner);border:1px solid var(--border2);'
            f'color:var(--text2);padding:7px 14px;border-radius:20px;font-size:12px;'
            f'cursor:pointer;transition:all .2s;" '
            f'onmouseover="this.style.borderColor=\'var(--orange)\';this.style.color=\'var(--orange)\'" '
            f'onmouseout="this.style.borderColor=\'var(--border2)\';this.style.color=\'var(--text2)\'"'
            f'>{q}</button>'
            for q in suggested_questions
        )
        sq_html = (
            f'<div style="margin-bottom:16px;">'
            f'<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
            f'text-transform:uppercase;color:var(--text3);margin-bottom:10px;">'
            f'Suggested questions</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{pills}</div>'
            f'</div>'
        )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{{
  --bg:#080e1c;--bg-card:#0d1628;--bg-inner:#0a1020;
  --orange:#f97316;--blue:#3b82f6;--teal:#06b6d4;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#475569;
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.14);
}}
.ib-assistant-wrap{{
  font-family:'Inter',sans-serif;background:var(--bg);
  padding:0 0 24px;
}}
.ib-topbar{{
  background:linear-gradient(135deg,#0d1628,#111827);
  border-bottom:1px solid var(--border2);padding:16px 28px;
  display:flex;align-items:center;gap:14px;
}}
.ib-topbar .badge{{
  background:var(--orange);color:#fff;width:32px;height:32px;
  border-radius:7px;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:12px;flex-shrink:0;
}}
.ib-topbar .titles h2{{font-size:16px;font-weight:700;color:var(--text);margin:0;}}
.ib-topbar .titles p{{font-size:12px;color:var(--text2);margin:2px 0 0;}}
.ib-topbar .ai-badge{{
  margin-left:auto;background:rgba(249,115,22,0.15);
  border:1px solid rgba(249,115,22,0.4);color:var(--orange);
  font-size:11px;font-weight:600;padding:4px 10px;border-radius:20px;
}}
.ib-context{{
  margin:24px 28px 0;background:var(--bg-card);
  border:1px solid var(--border2);border-radius:12px;padding:20px 22px;
  border-left:3px solid var(--orange);font-size:13px;
  color:var(--text2);line-height:1.65;
}}
.ib-chat-wrap{{
  margin:20px 28px 0;background:var(--bg-card);
  border:1px solid var(--border);border-radius:12px;overflow:hidden;
}}
.ib-chat-header{{
  background:rgba(255,255,255,0.03);padding:12px 18px;
  font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text2);border-bottom:1px solid var(--border);
}}
.ib-chat-body{{
  min-height:120px;padding:16px 18px;
  color:var(--text3);font-size:13px;font-style:italic;
  text-align:center;display:flex;align-items:center;justify-content:center;
}}
.ib-input-area{{margin:0 28px;margin-top:16px;}}
</style>
<div class="ib-assistant-wrap">
  <div class="ib-topbar">
    <div class="badge">IB</div>
    <div class="titles">
      <h2>{project_title}</h2>
      <p>{project_subtitle}</p>
    </div>
    <span class="ai-badge">AI Assistant</span>
  </div>
  <div class="ib-context">{blurb_html}</div>
  <div class="ib-chat-wrap">
    <div class="ib-chat-header">Conversation</div>
    <div class="ib-chat-body" id="ib-chat-output">
      Write a question or click a suggestion
    </div>
  </div>
  <div class="ib-input-area">
    {sq_html}
  </div>
</div>
"""


# ─────────────────────────────────────────────
#  GRADIO THEME KWARGS  (pass to gr.Blocks or gr.Interface)
# ─────────────────────────────────────────────
def get_gradio_theme():
    """
    Returns a dict of Gradio theme constructor kwargs that approximate the IB style.

    Fonts are wrapped in gr.themes.GoogleFont objects — recent Gradio versions
    compare theme fonts via Font.__eq__ on launch, which requires Font objects
    (passing bare strings raises: 'str' object has no attribute 'name').

    Usage:
        import gradio as gr
        from ib_style import get_gradio_theme
        theme = gr.themes.Base(**get_gradio_theme())
        with gr.Blocks(theme=theme, css=get_gradio_css()) as demo: ...
    """
    import gradio as gr
    return dict(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "ui-monospace"],
    )

def get_gradio_css():
    """Returns CSS string to inject into gr.Blocks for IB brand styling."""
    return """
    body, .gradio-container { background:#080e1c !important; color:#e2e8f0 !important;
                               font-family:'Inter',sans-serif !important; }
    .gradio-container { max-width:1200px !important; margin:0 auto !important; }

    /* Header strip */
    #component-0 > .wrap { border-bottom:1px solid rgba(255,255,255,0.08); }

    /* Tabs */
    .tab-nav { background:#0d1628 !important; border-bottom:1px solid rgba(255,255,255,0.08) !important; }
    .tab-nav button { color:#94a3b8 !important; font-size:13px !important;
                      font-weight:500 !important; padding:10px 20px !important; }
    .tab-nav button.selected { color:#f97316 !important;
                                border-bottom:2px solid #f97316 !important; }

    /* Panels / blocks */
    .block, .form { background:#0d1628 !important;
                     border:1px solid rgba(255,255,255,0.07) !important;
                     border-radius:12px !important; }

    /* Labels */
    label span, .block > label { color:#e2e8f0 !important; font-weight:600 !important;
                                  font-size:12px !important; }

    /* Inputs */
    input, select, textarea {
      background:#0a1020 !important; color:#e2e8f0 !important;
      border:1px solid rgba(255,255,255,0.14) !important;
      border-radius:8px !important; font-size:13px !important;
    }
    input:focus, select:focus, textarea:focus {
      border-color:#f97316 !important; outline:none !important;
      box-shadow:0 0 0 3px rgba(249,115,22,0.15) !important;
    }

    /* Sliders */
    input[type=range] { accent-color:#3b82f6 !important; }
    .svelte-slider .handle { background:#3b82f6 !important; }

    /* Primary button */
    button.primary, .btn-primary, #predict-btn {
      background:linear-gradient(135deg,#f97316,#ea580c) !important;
      color:#fff !important; font-weight:600 !important;
      border-radius:8px !important; border:none !important;
      padding:10px 24px !important; letter-spacing:.02em !important;
      transition:opacity .2s !important;
    }
    button.primary:hover { opacity:.88 !important; }

    /* Dropdown selected */
    .dropdown-arrow { color:#94a3b8 !important; }

    /* Plot outputs */
    .plot-container { background:#0a1020 !important;
                       border:1px solid rgba(255,255,255,0.07) !important;
                       border-radius:12px !important; padding:8px !important; }
    """


# ─────────────────────────────────────────────
#  SEABORN HELPER
# ─────────────────────────────────────────────
def apply_seaborn_style():
    """
    Apply IB style via seaborn's set_theme (call after apply_style()).
    Requires seaborn to be installed.
    """
    try:
        import seaborn as sns
        sns.set_theme(
            style="dark",
            palette=CYCLE,
            rc={
                "axes.facecolor":  MPL["bg_ax"],
                "figure.facecolor": MPL["bg"],
                "axes.edgecolor":  MPL["grid"],
                "grid.color":      MPL["grid"],
                "text.color":      MPL["text"],
                "axes.labelcolor": MPL["text2"],
                "xtick.color":     MPL["text2"],
                "ytick.color":     MPL["text2"],
                "axes.spines.top":   False,
                "axes.spines.right": False,
            }
        )
        print("✓ Seaborn IB theme applied.")
    except ImportError:
        print("seaborn not installed – skipping sns theme.")


# ─────────────────────────────────────────────
#  QUICK DEMO  (run this file directly to preview)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    apply_style()
    apply_seaborn_style()

    # Demo figure
    fig, axes = styled_fig(1, 2, figsize=(12, 4),
                            title="IB Style Demo",
                            subtitle="Brand palette applied to matplotlib")
    ax1, ax2 = axes

    # Bar chart
    cats = ["functional", "needs repair", "non functional"]
    vals = [33956, 4317, 22824]
    colours = [MPL["functional"], MPL["needs_repair"], MPL["non_functional"]]
    bars = ax1.bar(cats, vals, color=colours, width=0.6, edgecolor=MPL["bg_ax"], linewidth=0.5)
    ax1.set_title("Pump Status Distribution", color=MPL["text"])
    ax1.set_ylabel("Count")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{v:,}", ha="center", va="bottom", fontsize=9,
                 color=MPL["text2"])

    # Line chart
    x = np.linspace(0, 4 * np.pi, 200)
    ax2.plot(x, np.sin(x), color=MPL["blue"], label="Model A")
    ax2.plot(x, np.cos(x), color=MPL["orange"], label="Model B")
    ax2.axhline(0, color=MPL["grid"], linewidth=0.8, linestyle="--")
    ax2.set_title("Threshold Tuning Curve")
    ax2.legend()

    plt.tight_layout(pad=2)
    plt.savefig("/tmp/ib_style_demo.png")
    print("Demo saved to /tmp/ib_style_demo.png")
