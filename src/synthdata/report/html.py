"""The self-contained offline report page (a jinja2 template held in this module)."""

from __future__ import annotations

import base64
import io
import json
from html import escape

import jinja2
from markupsafe import Markup
import vl_convert as vlc
from PIL import Image

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.summary import twin_status

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<link rel="icon" href="{{ favicon }}">
<style>
:root { --fg:#1f2430; --muted:#6b7280; --bg:#f4f5f7; --card:#fff; --line:#e3e5ea; }
* { box-sizing: border-box; }
html, body { margin:0; overflow-x:hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color:var(--fg); background:var(--bg); line-height:1.45; }
main { max-width:1200px; margin:0 auto; padding:24px 16px 64px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px; margin:0 0 20px; }
h1 { margin:0 0 4px; font-size:1.6rem; } h2 { margin:0 0 6px; font-size:1.25rem; }
.source { color:var(--muted); font-size:.85rem; word-break:break-all; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.chip { background:#eef0f4; border-radius:999px; padding:3px 12px; font-size:.85rem; }
.metrics { display:flex; flex-wrap:wrap; gap:12px 28px; font-weight:600; }
.metrics span { white-space:nowrap; }
.muted { color:var(--muted); } .ok { color:#1a7f37; } .bad { color:#b42318; }
nav ol { margin:0; padding-left:20px; columns:2; } nav a { color:#2456a6; text-decoration:none; }
.caption { color:var(--muted); margin:0 0 14px; }
.grid { display:flex; flex-wrap:wrap; gap:22px 28px; align-items:flex-start; }
.grid.multiples { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:18px 20px; }
.grid.tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:14px; }
.fig { max-width:100%; overflow-x:auto; }
.tile { margin:0; } .tile figcaption { font-size:.85rem; font-weight:600; margin-top:4px; }
.tiles { margin-top:22px; }
.fig .vega-embed { max-width:100%; }
.links { font-size:.78rem; color:var(--muted); margin-top:4px; }
.links a { color:#2456a6; margin-right:8px; }
.tables { margin-top:12px; font-size:.85rem; }
.tables a { color:#2456a6; }
.tables .t { white-space:nowrap; margin-right:14px; }
.notes { color:var(--muted); font-size:.85rem; margin-top:10px; }
.tile img { display:block; width:100%; height:auto; border:1px solid var(--line); }
@media (max-width:640px) {
  main { padding:16px 16px 48px; } nav ol { columns:1; } .card { padding:16px; }
  .grid.tiles { grid-template-columns:repeat(2, 1fr); }
  /* on a phone a chart scales down to the card instead of scrolling sideways */
  .vega svg { max-width:100%; height:auto; }
}
</style>
</head>
<body>
<main>
<section class="card" id="summary">
<h1>{{ title }}</h1>
<div class="source">{{ source }}</div>
<div class="chips">
<span class="chip">{{ n_conditions }} conditions</span>
<span class="chip">{{ n_wells }} wells</span>
<span class="chip">{{ n_objects }} objects</span>
<span class="chip">reference: {{ reference }}</span>
</div>
<div class="metrics">
<span>clusters recover conditions: ARI {{ ari }}{% if ari_merged != 'n/a' %} (controls merged {{ ari_merged }}){% endif %}</span>
<span class="{{ {'ok': 'ok', 'flagged': 'bad'}.get(twin_kind, 'muted') }}">{{ twin_text }}</span>
<span>mean mAP (distinct conditions) {{ mean_map }}</span>
</div>
</section>
<nav class="card" id="toc"><strong>Contents</strong>
<ol>{% for s in sections %}<li><a href="#{{ s.key }}">{{ s.title }}</a></li>{% endfor %}</ol>
</nav>
{% for s in sections %}
<section class="card" id="{{ s.key }}">
<h2>{{ s.title }}</h2>
<p class="caption">{{ s.caption }}</p>
<div class="grid{{ ' multiples' if s.key == 'heatmaps' else '' }}">
{% for f in s.figures if f.spec %}
<div class="fig">
<div class="vega" aria-label="{{ f.title }}" id="v-{{ s.key }}-{{ f.name }}"></div>
<script type="application/json" class="vspec" data-target="v-{{ s.key }}-{{ f.name }}">{{ f.json }}</script>
<div class="links"><a href="{{ f.svg }}">svg</a><a href="{{ f.png }}">png</a><a href="{{ f.spec }}">spec</a></div></div>
{% endfor %}
</div>
{% if s.figures | rejectattr('spec') | list %}
<div class="grid tiles">
{% for f in s.figures if not f.spec %}
<figure class="tile"><a href="{{ f.png }}"><img alt="{{ f.title }}" src="{{ f.uri }}"></a><figcaption>{{ f.title }} <span class="links"><a href="{{ f.png }}">full size</a></span></figcaption></figure>
{% endfor %}
</div>
{% endif %}
{% if s.tables %}<div class="tables">Tables: {% for t in s.tables %}<span class="t">{{ t.name }} (<a href="{{ t.parquet }}">parquet</a> · <a href="{{ t.csv }}">csv</a>)</span> {% endfor %}</div>{% endif %}
{% for n in s.notes %}<div class="notes">{{ n }}</div>{% endfor %}
</section>
{% endfor %}
</main>
<script>{{ bundle }}</script>
<script>
document.querySelectorAll("script.vspec").forEach(function (s) {
  var el = document.getElementById(s.dataset.target);
  vegaEmbed(el, JSON.parse(s.textContent), {actions: {export: true, source: false, compiled: false, editor: false}, renderer: "svg"})
    .catch(function (e) { el.textContent = "figure failed to render: " + e; });
});
</script>
</body>
</html>
"""

# a 96-well plate glyph, inline so the offline page makes no favicon request
FAVICON = ("data:image/svg+xml," + "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
           "%3Crect width='16' height='16' rx='3' fill='%234e79a7'/%3E"
           + "".join(f"%3Ccircle cx='{x}' cy='{y}' r='1.3' fill='white'/%3E" for y in (4, 8, 12) for x in (4, 8, 12))
           + "%3C/svg%3E")

_ENV = jinja2.Environment(autoescape=True, undefined=jinja2.StrictUndefined)


def _safe_json(spec: dict) -> str:
    text = json.dumps(spec, separators=(",", ":"), sort_keys=True)
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


THUMB_PX = 384


def _thumb_uri(path) -> str:
    """The tile as a PNG data URI of at most THUMB_PX on a side; the full-size file stays on disk, linked."""
    with Image.open(path) as im:
        im = im.copy()
    im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def render_html(ctx: ReportContext, results: list[SectionResult], metrics: dict) -> str:
    root = ctx.root
    secs = []
    for r in results:
        figs = []
        for f in r.figures:
            d = {"name": f.name, "title": f.title, "spec": f.spec, "svg": f.svg, "png": f.png}
            if f.spec:
                spec = json.loads((root / f.spec).read_text())
                d["json"] = Markup(_safe_json(spec))
            else:
                d["uri"] = _thumb_uri(root / f.png)
            figs.append(d)
        secs.append({"key": r.key, "title": r.title, "caption": r.caption, "figures": figs,
                     "tables": [t.__dict__ for t in r.tables], "notes": r.notes})
    twin_kind, twin_text = twin_status(metrics)
    return _ENV.from_string(TEMPLATE).render(
        title=ctx.inputs.title, source=ctx.inputs.source, n_conditions=len(ctx.conditions),
        n_wells=metrics["n_wells"], n_objects=metrics["n_objects"], reference=ctx.inputs.reference,
        ari=_fmt(metrics.get("ari")), mean_map=_fmt(metrics.get("mean_map")),
        twin_kind=twin_kind, twin_text=twin_text, ari_merged=_fmt(metrics.get("ari_control_merged")),
        sections=secs, favicon=FAVICON,
        bundle=Markup(vlc.javascript_bundle()))
