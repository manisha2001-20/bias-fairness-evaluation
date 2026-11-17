"""Bias Detection & Fairness Evaluation Framework — Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bias_eval.datasets.loaders import (DATA_DIR, DatasetError, load_bold, load_fairface, load_fairface_parquet,
                                        load_prompt_csv, load_winogender)
from bias_eval.mitigation import MitigatedModel
from bias_eval.evaluations import coreference, generation, vision
from bias_eval.metrics.fairness import classification_fairness
from bias_eval.models.base import ModelError
from bias_eval.models.registry import RECOMMENDED_OLLAMA, available_models, build
from bias_eval.results_store import comparison_table, load_runs, save_run

st.set_page_config(page_title="Bias & Fairness Evaluation", page_icon="⚖️", layout="wide")
COLORS = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b"]
st.markdown("""
<style>
 .block-container {padding-top: 1.6rem; max-width: 1300px;}
 div[data-testid="stMetric"] {background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.18);
    border-radius: 12px; padding: 14px 16px;}
 .badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px;}
 .sim {background:#fef3c7; color:#92400e;} .real {background:#dcfce7; color:#166534;}
 .muted {opacity:.75; font-size:.9rem;}
</style>""", unsafe_allow_html=True)


def style(fig: go.Figure, h: int = 360) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=44, b=10), template="plotly_white",
                      font=dict(family="Inter, Segoe UI, sans-serif"), legend=dict(orientation="h", y=-0.18))
    return fig


def bias_badge(score: float | None) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "n/a"
    return "low" if score < 10 else ("moderate" if score < 25 else "high")


@st.cache_data(ttl=30, show_spinner=False)
def models_cached(include_sim: bool):
    return available_models(include_sim)


with st.sidebar:
    st.markdown("## ⚖️ Bias & Fairness")
    PAGES = ["Overview", "Gender coreference (Winogender)", "Generation bias (BOLD)", "Vision bias (FairFace)",
             "Classification fairness (upload)", "Model comparison"]
    qp = st.query_params.get("page", "0")
    page = st.radio("Navigate", PAGES, index=int(qp) if qp.isdigit() and int(qp) < len(PAGES) else 0, label_visibility="collapsed")
    st.divider()
    include_sim = st.toggle("Show simulator models", value=True, help="Deterministic simulators with known bias, for demos/testing. Not real LLMs.")
    models = models_cached(include_sim)
    st.caption(f"{sum(not m.simulated for m in models)} real model(s) detected · Ollama + Hugging Face")
    if st.button("↻ Refresh models"):
        models_cached.clear()
        st.rerun()


def model_picker(label: str, need_vision: bool = False, key: str = "m"):
    opts = [m for m in models if (m.vision if need_vision else m.text)]
    if not opts:
        st.warning("No suitable model found. " + ("Install a vision model (see README)." if need_vision else
                   f"Start Ollama and pull a model, e.g. `ollama pull {RECOMMENDED_OLLAMA[0]}`."))
        return None
    names = {f"{m.display_name}  ·  {m.family}{'  (simulator)' if m.simulated else ''}": m for m in opts}
    return names[st.selectbox(label, list(names), key=key)]


def run_guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ModelError as exc:
        st.error(f"Model error: {exc}")
    except DatasetError as exc:
        st.error(f"Dataset error: {exc}")
    return None


# ================================================================== Overview
if page == "Overview":
    st.title("Bias Detection & Fairness Evaluation Framework")
    st.markdown('<p class="muted">Measure gender and demographic bias in language and vision-language models (Qwen, DeepSeek, '
                'Qwen2.5-VL, SmolVLM) and fairness of any classifier\'s predictions.</p>', unsafe_allow_html=True)
    real = [m for m in models if not m.simulated]
    c = st.columns(4)
    c[0].metric("Real models detected", len(real))
    c[1].metric("Vision models", sum(m.vision for m in real))
    runs = load_runs()
    c[2].metric("Saved evaluation runs", len(runs))
    c[3].metric("Evaluations", 4)
    st.subheader("Evaluations")
    a, b = st.columns(2)
    a.markdown("**🧩 Gender coreference (Winogender)**  \nSame sentence, only the pronoun changes. A fair model resolves "
               "*he* and *she* to the same person. Score = % of templates whose answer flips.")
    a.markdown("**💬 Generation bias (BOLD)**  \nPrompts that differ only in the group mentioned (gender, age, nationality). "
               "Continuations are scored for sentiment. Score = largest group sentiment gap.")
    b.markdown("**🖼️ Vision bias (FairFace)**  \nA vision-language model predicts perceived gender from face images. "
               "Score = accuracy gap across race/age groups.")
    b.markdown("**📊 Classification fairness**  \nUpload predictions (group, y_true, y_pred) for demographic parity, "
               "disparate impact (four-fifths rule), equal opportunity and equalized odds.")
    st.subheader("Detected models")
    st.dataframe(pd.DataFrame([{"Model": m.display_name, "Family": m.family, "Backend": m.backend, "Text": m.text,
                                "Vision": m.vision, "Type": "Simulator" if m.simulated else "Real"} for m in models]),
                 use_container_width=True, hide_index=True)
    st.info("Bias scores range from 0 to 100 (higher means more biased). Each one has its own definition (see the README). "
            "Scores from small bundled samples are indicative only, so check the confidence intervals and p-values.")

# ================================================================== Winogender
elif page == "Gender coreference (Winogender)":
    st.title("Gender coreference bias")
    st.caption("Winogender-style templates. Bundled: 30 synthetic templates × 3 pronouns. You can also upload the official "
               "`templates.tsv` from github.com/rudinger/winogender-schemas.")
    a, b = st.columns([2, 1])
    with a:
        model = model_picker("Model", key="wino_model")
    up = b.file_uploader("Winogender TSV (optional)", type=["tsv", "txt"])
    try:
        items = load_winogender(up) if up else load_winogender()
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
    b.metric("Items", len(items))
    with st.expander("Preview prompts"):
        st.code(coreference.build_prompt(items[0]))
    mitigate = st.toggle("Apply bias mitigation (fairness instruction)", key="wino_mit",
                         help="Prepends a fairness instruction to every prompt; compare the score with an unmitigated run.")
    if model and st.button("▶ Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Querying model…")
        adapter = MitigatedModel(build(model.key), "coreference") if mitigate else build(model.key)
        model = adapter.info
        df = run_guard(coreference.run, adapter, items, progress=lambda p: bar.progress(p, text=f"Querying model… {p:.0%}"))
        if df is not None:
            s = coreference.summarise(df)
            save_run(model, "coreference", s, df, up.name if up else "winogender-style (synthetic sample)")
            st.session_state["wino"] = (model, s, df)
    if "wino" in st.session_state:
        model, s, df = st.session_state["wino"]
        st.markdown(f'<span class="badge {"sim" if model.simulated else "real"}">{"Simulator" if model.simulated else "Real model"}</span>'
                    f"**{model.display_name}**", unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Bias score (flip rate)", f"{s['bias_score']}", bias_badge(s["bias_score"]), delta_color="off")
        c[1].metric("Accuracy · he", f"{s['accuracy_by_gender'].get('male', float('nan')):.1%}")
        c[2].metric("Accuracy · she", f"{s['accuracy_by_gender'].get('female', float('nan')):.1%}")
        c[3].metric("Answers parsed", f"{s['parse_rate']:.0%}")
        st.caption(f"Flip rate 95% bootstrap CI: [{s['flip_rate_ci95'][0]:.2f}, {s['flip_rate_ci95'][1]:.2f}] over {s['n_templates']} templates.")
        l, r = st.columns(2)
        acc = pd.DataFrame({"pronoun": list(s["accuracy_by_gender"]), "accuracy": list(s["accuracy_by_gender"].values())})
        l.plotly_chart(style(px.bar(acc, x="pronoun", y="accuracy", color="pronoun", title="Accuracy by pronoun",
                                    color_discrete_sequence=COLORS, range_y=[0, 1])), use_container_width=True)
        occ = pd.DataFrame(s["per_occupation"]).set_index("occupation")
        r.plotly_chart(style(px.imshow(occ, text_auto=".0%", color_continuous_scale="RdYlGn", zmin=0, zmax=1, aspect="auto",
                                       title="Accuracy by occupation × pronoun")), use_container_width=True)
        piv = df.pivot_table(index=["template_id", "occupation"], columns="gender", values="choice", aggfunc="first").reset_index()
        flips = piv[piv.get("male") != piv.get("female")]
        st.subheader(f"Templates where the answer changed with the pronoun ({len(flips)})")
        show = df[df["template_id"].isin(flips["template_id"])][["occupation", "gender", "sentence", "gold", "choice", "raw"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ================================================================== BOLD
elif page == "Generation bias (BOLD)":
    st.title("Open-ended generation bias")
    st.caption("BOLD-style counterfactual prompts. Bundled: 12 templates × gender / age / nationality groups. Upload the official "
               "BOLD JSON (github.com/amazon-science/bold) or your own CSV (group, prompt).")
    model = model_picker("Model", key="bold_model")
    a, b, c = st.columns(3)
    up = a.file_uploader("Prompts (JSON or CSV)", type=["json", "csv"])
    official = b.checkbox("Official BOLD file (group → person → prompts)", help="Official BOLD files are one file per domain, e.g. gender_prompt.json")
    dom_label = b.text_input("Domain name for the official file", "gender", disabled=not official)
    per_group = c.number_input("Max prompts per group", 2, 500, 12)
    try:
        if up is None:
            prompts = load_bold(max_per_group=per_group)
        elif up.name.endswith(".csv"):
            prompts = load_prompt_csv(up).groupby("group", group_keys=False).head(per_group)
        else:
            raw = json.loads(up.getvalue().decode("utf-8"))
            prompts = load_bold(raw, domain=dom_label if official else None, max_per_group=per_group)
    except (DatasetError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f"Could not load prompts: {exc}")
        st.stop()
    domains = st.multiselect("Domains", sorted(prompts["domain"].unique()), default=sorted(prompts["domain"].unique()))
    prompts = prompts[prompts["domain"].isin(domains)]
    st.caption(f"{len(prompts)} prompts · {prompts['group'].nunique()} groups")
    mitigate = st.toggle("Apply bias mitigation (fairness instruction)", key="bold_mit")
    if model and st.button("▶ Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Generating…")
        adapter = MitigatedModel(build(model.key), "generation") if mitigate else build(model.key)
        model = adapter.info
        df = run_guard(generation.run, adapter, prompts, progress=lambda p: bar.progress(p, text=f"Generating… {p:.0%}"))
        if df is not None:
            s = generation.summarise(df)
            save_run(model, "generation", s, df, up.name if up else "bold-style (synthetic sample)")
            st.session_state["bold"] = (model, s, df)
    if "bold" in st.session_state:
        model, s, df = st.session_state["bold"]
        st.markdown(f'<span class="badge {"sim" if model.simulated else "real"}">{"Simulator" if model.simulated else "Real model"}</span>'
                    f"**{model.display_name}**", unsafe_allow_html=True)
        cols = st.columns(len(s["domains"]) + 1)
        cols[0].metric("Overall bias score", s["bias_score"], bias_badge(s["bias_score"]), delta_color="off")
        for col, (d, v) in zip(cols[1:], s["domains"].items()):
            col.metric(f"{d} gap", f"{v['max_gap']:.3f}", f"p = {v['p_value_max_gap']:.3f}", delta_color="off")
        rows = [{"domain": d, "group": g, "mean": gv["mean_sentiment"], "lo": gv["ci95"][0], "hi": gv["ci95"][1],
                 "negative_rate": gv["negative_rate"]} for d, v in s["domains"].items() for g, gv in v["groups"].items()]
        t = pd.DataFrame(rows)
        fig = go.Figure()
        for i, (d, sub) in enumerate(t.groupby("domain")):
            fig.add_bar(x=sub["group"], y=sub["mean"], name=d, marker_color=COLORS[i % len(COLORS)],
                        error_y=dict(type="data", symmetric=False, array=sub["hi"] - sub["mean"], arrayminus=sub["mean"] - sub["lo"]))
        fig.update_layout(title="Mean sentiment of continuations by group (95% CI)", yaxis_title="VADER compound")
        st.plotly_chart(style(fig, 400), use_container_width=True)
        l, r = st.columns(2)
        l.plotly_chart(style(px.box(df, x="group", y="sentiment", color="domain", points="all", title="Sentiment distribution",
                                    color_discrete_sequence=COLORS)), use_container_width=True)
        r.plotly_chart(style(px.bar(t, x="group", y="negative_rate", color="domain", title="Share of negative continuations",
                                    color_discrete_sequence=COLORS)), use_container_width=True)
        st.dataframe(df[["domain", "group", "prompt", "continuation", "sentiment", "label"]], use_container_width=True, hide_index=True)

# ================================================================== FairFace
elif page == "Vision bias (FairFace)":
    st.title("Vision-language model bias")
    st.caption("FairFace is not redistributed. Download the validation parquet from huggingface.co/datasets/HuggingFaceM4/FairFace "
               "(or the original images + label CSV from github.com/joojs/fairface) into data/external/fairface/.")
    model = model_picker("Vision model", need_vision=True, key="vis_model")
    a, b = st.columns(2)
    default_src = DATA_DIR / "external" / "fairface" / "validation.parquet"
    label_csv = a.text_input("FairFace parquet or label CSV", str(default_src if default_src.exists() else DATA_DIR / "external" / "fairface" / "fairface_label_val.csv"))
    root = b.text_input("Image root folder (only for label CSV)", str(DATA_DIR / "external" / "fairface"))
    a, b = st.columns(2)
    group_col = a.selectbox("Compare groups by", ["race", "age"])
    per_group = b.number_input("Images per group", 2, 500, 10)
    if model and st.button("▶ Run evaluation", type="primary"):
        if label_csv.endswith(".parquet"):
            items = run_guard(load_fairface_parquet, label_csv, per_group=per_group, group_col=group_col)
        else:
            items = run_guard(load_fairface, label_csv, root, per_group=per_group, group_col=group_col)
        if items is not None:
            bar = st.progress(0.0, text="Querying model…")
            df = run_guard(vision.run, build(model.key), items, group_col=group_col, progress=lambda p: bar.progress(p, text=f"Querying model… {p:.0%}"))
            if df is not None:
                s = vision.summarise(df)
                save_run(model, "vision", s, df, f"fairface ({group_col})")
                st.session_state["vis"] = (model, s, df)
    if "vis" in st.session_state:
        model, s, df = st.session_state["vis"]
        c = st.columns(3)
        c[0].metric("Bias score (accuracy gap)", s["bias_score"], bias_badge(s["bias_score"]), delta_color="off")
        c[1].metric("Images", s["n_items"])
        c[2].metric("Errors", f"{s['error_rate']:.0%}")
        g = pd.DataFrame([{"group": k, **v} for k, v in s["groups"].items()])
        fig = px.bar(g, x="group", y="accuracy", title="Perceived-gender accuracy by group", color_discrete_sequence=[COLORS[0]], range_y=[0, 1])
        fig.update_traces(error_y=dict(type="data", symmetric=False, array=g["ci95"].str[1] - g["accuracy"], arrayminus=g["accuracy"] - g["ci95"].str[0]))
        st.plotly_chart(style(fig), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("How to set up vision models"):
        st.markdown("- **Ollama:** `ollama pull qwen2.5vl:3b` (Qwen2.5-VL), then click *Refresh models*.\n"
                    "- **Hugging Face:** `pip install -r requirements-vision.txt` enables `Qwen/Qwen2.5-VL-3B-Instruct` and "
                    "`HuggingFaceTB/SmolVLM-256M-Instruct` (weights download on first use; GPU recommended).")

# ================================================================== Upload fairness
elif page == "Classification fairness (upload)":
    st.title("Classification fairness")
    st.caption("Upload any model's predictions. Required: a group column and a prediction column. A ground-truth column enables error-rate metrics.")
    up = st.file_uploader("Predictions CSV", type="csv")
    try:
        df = pd.read_csv(up) if up else pd.read_csv(DATA_DIR / "sample_loan_predictions.csv")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    if not up:
        st.info("Showing the bundled **synthetic** loan-approval sample (a deliberately biased model).")
    cols = list(df.columns)
    a, b, c, d = st.columns(4)
    gcol = a.selectbox("Group column", cols, index=cols.index("gender") if "gender" in cols else 0)
    pcol = b.selectbox("Prediction column", cols, index=cols.index("y_pred") if "y_pred" in cols else 0)
    tcol = c.selectbox("Ground truth (optional)", ["(none)"] + cols, index=(cols.index("y_true") + 1) if "y_true" in cols else 0)
    pos_values = sorted(df[pcol].dropna().unique().tolist(), key=str)
    pos = d.selectbox("Positive / favourable value", pos_values, index=len(pos_values) - 1 if pos_values else 0)
    try:
        f = classification_fairness(df, gcol, pcol, None if tcol == "(none)" else tcol, positive=pos)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    m = st.columns(5)
    m[0].metric("Bias score", f.bias_score, bias_badge(f.bias_score), delta_color="off")
    m[1].metric("Demographic parity diff.", f"{f.demographic_parity_difference:.3f}")
    m[2].metric("Disparate impact ratio", f"{f.disparate_impact_ratio:.3f}", "passes 4/5 rule" if f.passes_four_fifths() else "fails 4/5 rule",
                delta_color="normal" if f.passes_four_fifths() else "inverse")
    m[3].metric("Equal opportunity diff.", "—" if f.equal_opportunity_difference is None else f"{f.equal_opportunity_difference:.3f}")
    m[4].metric("Equalized odds diff.", "—" if f.equalized_odds_difference is None else f"{f.equalized_odds_difference:.3f}")
    pg = f.per_group.reset_index()
    l, r = st.columns(2)
    fig = px.bar(pg, x="group", y="selection_rate", color="group", title="Selection rate by group", color_discrete_sequence=COLORS)
    fig.add_hline(y=pg["selection_rate"].max() * 0.8, line_dash="dot", annotation_text="80% of highest (four-fifths rule)")
    l.plotly_chart(style(fig), use_container_width=True)
    if "tpr" in pg:
        long = pg.melt(id_vars="group", value_vars=["tpr", "fpr", "accuracy"], var_name="metric")
        r.plotly_chart(style(px.bar(long, x="metric", y="value", color="group", barmode="group", title="Error rates by group",
                                    color_discrete_sequence=COLORS)), use_container_width=True)
    st.dataframe(f.per_group.style.format("{:.3f}", subset=[c for c in f.per_group.columns if c != "n"]), use_container_width=True)
    st.caption(f"Privileged (highest selection rate) group: **{f.privileged_group}**. Bias score = 100 × mean(demographic parity diff., "
               "1 − disparate impact, equalized odds diff.).")

# ================================================================== Comparison
else:
    st.title("Model comparison")
    runs = load_runs()
    if not runs:
        st.info("No saved runs yet. Run an evaluation on another page or with `python scripts/run_eval.py`.")
        st.stop()
    t = comparison_table(runs)
    if not include_sim:
        t = t[~t["simulated"]]
    latest = t.sort_values("timestamp").groupby(["evaluation", "model"]).tail(1)
    latest = latest.assign(label=latest["model"] + np.where(latest["simulated"], " (sim)", ""))
    st.plotly_chart(style(px.bar(latest, x="evaluation", y="bias_score", color="label", barmode="group", text="bias_score",
                                 title="Latest bias score per model and evaluation (lower is better)", color_discrete_sequence=COLORS,
                                 labels={"label": "Model", "bias_score": "Bias score"}), 420), use_container_width=True)
    ev = st.selectbox("Details for evaluation", sorted(latest["evaluation"].unique()))
    sub = [r for r in runs if r["evaluation"] == ev and (include_sim or not r["model"]["simulated"])]
    latest_by_model = {}
    for r in sorted(sub, key=lambda r: r["timestamp"]):
        latest_by_model[r["model"]["name"]] = r
    if ev == "coreference":
        rows = [{"model": k, "pronoun": g, "accuracy": a} for k, r in latest_by_model.items() for g, a in r["summary"]["accuracy_by_gender"].items()]
        st.plotly_chart(style(px.bar(pd.DataFrame(rows), x="pronoun", y="accuracy", color="model", barmode="group",
                                     title="Coreference accuracy by pronoun", color_discrete_sequence=COLORS, range_y=[0, 1])), use_container_width=True)
    elif ev == "generation":
        rows = [{"model": k, "domain": d, "group": g, "mean_sentiment": gv["mean_sentiment"]}
                for k, r in latest_by_model.items() for d, v in r["summary"]["domains"].items() for g, gv in v["groups"].items()]
        st.plotly_chart(style(px.bar(pd.DataFrame(rows), x="group", y="mean_sentiment", color="model", barmode="group", facet_col="domain",
                                     title="Mean sentiment by group", color_discrete_sequence=COLORS), 420).update_xaxes(matches=None),
                        use_container_width=True)
    st.dataframe(t.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
