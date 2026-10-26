"""
'LesionNet vs the classic models' section for the Model Comparison page.
Reads the result files written by research/lesion_net/evaluate.py and eval_external.py.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "research", "lesion_net", "results")
BASELINES = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]
NAMES = {"LesionNet_A3_full": "LesionNet", "LesionNet_A0_plain": "Plain EfficientNet (A0)",
         "LesionNet_A5_mobilenet": "LesionNet-Mobile (A5)"}


def _read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    # A0 has no trained segmentation head (λ_seg = 0), so its "lesion map" means nothing
    a0 = df["Model"] == "LesionNet_A0_plain"
    for col in ("e_lesion", "e_healthy", "e_bg", "focus_ratio", "pointing_hit", "iou_05", "severity_spearman",
                "severity_mae_pct", "lesion_pct_on_healthy", "lesion_pct_healthy", "lesion_pct_diseased"):
        if col in df.columns:
            df.loc[a0, col] = float("nan")
    df["Model"] = df["Model"].map(lambda m: NAMES.get(m, m.replace("LesionNet_", "LesionNet ")))
    return df


def _color(m):
    if m.startswith("LesionNet-Mobile"):
        return "#5EAAA8"
    if m.startswith("LesionNet"):
        return theme.MODEL_COLORS.get("LesionNet", theme.GREEN)
    return theme.MODEL_COLORS.get(m, "#9AA59C")


def render():
    ev = _read("eval_summary.csv")
    pdoc = _read("plantdoc_summary.csv")
    if ev is None:
        return
    st.markdown("### 🛡️ LesionNet vs the classic models")
    st.caption("LesionNet (our model) first outlines the diseased spots, then names the disease from those "
               "spots only. The tests below change the photo's background or use real field photos, "
               "to check whether a model really looks at the disease.")

    ln = ev[ev["Model"] == "LesionNet"]
    base = ev[ev["Model"].isin(BASELINES)]
    if len(ln) and len(base):
        ln = ln.iloc[0]
        best = lambda col: base[col].max()  # noqa: E731
        c = st.columns(4)
        c[0].metric("Accuracy, normal photos", f"{ln['acc_clean']:.1%}",
                    f"{(ln['acc_clean'] - best('acc_clean')) * 100:+.1f} pts vs best classic")
        c[1].metric("Accuracy, grey background", f"{ln['acc_grey_bg']:.1%}",
                    f"{(ln['acc_grey_bg'] - best('acc_grey_bg')) * 100:+.1f} pts vs best classic")
        c[2].metric("Evidence on diseased spots", f"{ln['e_lesion']:.0%}",
                    f"{(ln['e_lesion'] - best('e_lesion')) * 100:+.0f} pts vs best classic")
        c[3].metric("Severity match (ρ)", f"{ln['severity_spearman']:.2f}", "classic heatmaps: ≤ 0.13")

    tests = [("acc_clean", "Normal photos"), ("acc_grey_bg", "Grey background"), ("acc_swap_bg", "Swapped background")]
    fig = go.Figure()
    for _, r in ev.iterrows():
        xs = [t[1] for t in tests]
        ys = [r[t[0]] for t in tests]
        if pdoc is not None and r["Model"] in set(pdoc["Model"]):
            xs.append("Field photos (PlantDoc)")
            ys.append(float(pdoc.loc[pdoc["Model"] == r["Model"], "acc12"].iloc[0]))
        fig.add_bar(name=r["Model"], x=xs, y=ys, marker_color=_color(r["Model"]),
                    text=[f"{v:.0%}" for v in ys], textposition="outside")
    fig.update_layout(barmode="group", template="agrix", height=420, yaxis_range=[0, 1.12],
                      yaxis_tickformat=".0%", title="Accuracy when the photo changes",
                      margin=dict(t=50, b=30), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        ex = ev.dropna(subset=["e_lesion"])
        fig2.add_bar(name="On diseased spots", x=ex["Model"], y=ex["e_lesion"], marker_color=theme.RED)
        fig2.add_bar(name="On the background", x=ex["Model"], y=ex["e_bg"], marker_color="#9AA59C")
        fig2.update_layout(barmode="group", template="agrix", height=360, yaxis_tickformat=".0%",
                           title="Where each model's evidence comes from", margin=dict(t=50, b=30),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        ex = ev.dropna(subset=["pointing_hit"])
        fig3 = go.Figure(go.Bar(x=ex["Model"], y=ex["pointing_hit"], marker_color=[_color(m) for m in ex["Model"]],
                                text=[f"{v:.0%}" for v in ex["pointing_hit"]], textposition="outside"))
        fig3.update_layout(template="agrix", height=360, yaxis_range=[0, 1.12], yaxis_tickformat=".0%",
                           title="Strongest evidence lands on a diseased spot", margin=dict(t=50, b=30))
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Full results and how to read them"):
        st.markdown("""
- **Grey / swapped background:** the same leaf is pasted onto a plain grey background or the background
  of a different photo. A model that uses the leaf should keep its accuracy.
- **Field photos (PlantDoc):** real photos from fields and the internet, never used for training; only the
  12 classes shared with our 15 are scored (chance 8 %). All models drop a lot here — lab photos are a
  poor teacher for the field.
- **Plain EfficientNet (A0):** LesionNet's exact backbone, data split and training schedule, but
  ordinary pooling and no lesion outlines. It is just as accurate on normal photos, so the robustness
  gain comes from LesionNet's design, not from a better backbone or more training.
- **LesionNet-Mobile (A5):** a smaller, phone-friendly LesionNet (MobileNetV2 backbone).
- **Evidence on spots:** classic models are explained with Grad-CAM++; LesionNet with its own lesion map.
  This test uses automatic lesion masks LesionNet was trained to imitate, so it favours LesionNet; the
  background and PlantDoc tests do not.
""")
        st.dataframe(ev, use_container_width=True, hide_index=True)
        if pdoc is not None:
            st.dataframe(pdoc, use_container_width=True, hide_index=True)
        fig_png = os.path.join(RESULTS, "fig_lesionnet_vs_baselines.png")
        if os.path.exists(fig_png):
            st.image(fig_png, width="stretch")
