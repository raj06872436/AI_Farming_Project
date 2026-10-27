"""
LesionNet research content for the Research Dashboard, Model Comparison and Reports pages.
Everything is read from files in the repo (results CSVs, reports/, paper/), so re-running the
evaluation scripts updates the site.
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lesionnet_inference as lni
import theme

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LN_RESULTS = os.path.join(BASE_DIR, "research", "lesion_net", "results")
LN_FIGS = os.path.join(LN_RESULTS, "figures")
STUDY_DIR = os.path.join(BASE_DIR, "research", "results")
REPORTS = os.path.join(BASE_DIR, "reports")
PAPER_DIR = os.path.join(BASE_DIR, "paper")
SAVED = os.path.join(BASE_DIR, "saved_models")

PAPER_TITLE = ("LesionNet: Lesion-Gated Classification for Background-Robust, Faithfully Explainable "
               "and Severity-Aware Plant Disease Diagnosis")
PAPER_AUTHORS = "Raj Naik, Dhruv Patel, Shreya Mehta, Parshva Mathukia and Prachi Tawde"
PAPER_AFFIL = "Department of Information Technology, Dwarkadas J. Sanghvi College of Engineering, Mumbai, India"
PAPER_ABSTRACT = (
    "Convolutional neural networks (CNNs) reach near-perfect accuracy on the PlantVillage benchmark, yet it is "
    "unclear whether they recognise the disease or the photographic setting in which each class was captured. "
    "On a held-out split of 3,109 images from 15 tomato, potato and pepper classes, four fine-tuned CNNs place "
    "only 7–12% of their Grad-CAM++ evidence on diseased tissue and 25–49% on the background, and the most "
    "accurate of them (ResNet50, 99.8%) falls to 45.1% when the same leaves are placed on a plain grey background. "
    "We propose LesionNet, a segment-then-classify network in which a segmentation head labels every pixel as "
    "background, healthy leaf or lesion, and the classifier receives only features pooled over the predicted "
    "lesion and leaf pixels plus the predicted lesion area (lesion-gated pooling). The gate is detached from the "
    "classification loss, so the lesion map is not a post-hoc explanation but the actual input to the decision, "
    "and it doubles as a severity estimate. LesionNet is trained without manual annotation, from colour-based "
    "pseudo-masks and background-swap augmentation. It matches the baselines on clean images (99.5%) while "
    "keeping 98.7% accuracy on grey backgrounds, against 67.5% for an identically trained plain EfficientNetB0; "
    "its evidence lands on a lesion in 97% of images (CNNs: 49–64%) and its predicted diseased area correlates "
    "with the measured area at ρ = 0.90. On 1,012 PlantDoc field photographs all models, LesionNet included, "
    "drop to 19.6–23.6%, showing that background robustness is necessary but not sufficient for field use.")
BIBTEX = """@inproceedings{naik2026lesionnet,
  title     = {LesionNet: Lesion-Gated Classification for Background-Robust,
               Faithfully Explainable and Severity-Aware Plant Disease Diagnosis},
  author    = {Naik, Raj and Patel, Dhruv and Mehta, Shreya and Mathukia, Parshva and Tawde, Prachi},
  year      = {2026},
  note      = {Manuscript in preparation}
}"""

NICE = {"LesionNet_A0_plain": "Plain EfficientNet (A0)", "LesionNet_A3_full": "LesionNet",
        "LesionNet_A5_mobilenet": "LesionNet-Mobile"}

# Extra Model Comparison entries for the two LesionNet models
LN_ARCH = {
    lni.NAME: {
        "architecture": "LesionNet (EfficientNetB0 backbone + lesion segmentation head + lesion-gated pooling)",
        "strengths": "Same accuracy as ResNet50 at a fifth of the parameters; keeps 98.7% when the background "
                     "changes; its lesion map is the real evidence and measures severity (ρ = 0.90).",
        "weaknesses": "Like every model here, weak on real field photos (≈21% on PlantDoc); the lesion map is "
                      "learned from automatic colour masks.",
        "deployment": "Default model in AGRI-X; PCs and laptops",
    },
    lni.MOBILE: {
        "architecture": "LesionNet-Mobile (MobileNetV2 backbone + lesion segmentation head + lesion-gated pooling)",
        "strengths": "Lightest robust model: 94.2% on a new background (classic MobileNetV2: 76.0%).",
        "weaknesses": "Training stopped at epoch 10/18 (GPU fault); 96.0% accuracy, a bit below LesionNet.",
        "deployment": "Phones and slower computers",
    },
}


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def lesionnet_metrics():
    """Full metrics (accuracy, precision, recall, F1, AUC) from reports/metrics/lesionnet_metrics.csv."""
    return _csv(os.path.join(REPORTS, "metrics", "lesionnet_metrics.csv"))


def research_models(registry):
    """Classic models + LesionNet models that have evaluation reports."""
    names = list(registry)
    ln = lesionnet_metrics()
    if ln is not None:
        names += [m for m in ln["Model"] if m not in names]
    return names


def compare_registry(registry):
    """REGISTRY plus LesionNet entries, with the same keys, for the Model Comparison charts."""
    out = {k: dict(v) for k, v in registry.items()}
    ln = lesionnet_metrics()
    for name in lni.available_models():
        meta = lni.METAS[name]
        row = ln[ln["Model"] == name].iloc[0] if ln is not None and name in set(ln["Model"]) else None
        out[name] = {
            "accuracy": float(row["Accuracy"]) if row is not None else meta["accuracy"],
            "params": meta["params"], "size_mb": float(row["Model Size (MB)"]) if row is not None else meta["size_mb"],
            "inference_ms": meta["inference_ms"], "status": "verified", **LN_ARCH[name],
        }
    return out


def metrics_table(metrics_df):
    """The classic models' metrics table with LesionNet rows appended."""
    ln = lesionnet_metrics()
    if metrics_df is None or ln is None:
        return metrics_df
    add = ln.copy()
    add["Inference Time (ms)"] = [lni.METAS.get(m, {}).get("inference_ms") for m in add["Model"]]
    add["Trainable Params"] = None
    cols = [c for c in metrics_df.columns if c in add.columns]
    return pd.concat([metrics_df, add[cols]], ignore_index=True)


# ───────────────────────────────────────────────────────── Research Dashboard tab
def render_study_tab():
    ev = _csv(os.path.join(LN_RESULTS, "eval_summary.csv"))
    pdoc = _csv(os.path.join(LN_RESULTS, "plantdoc_summary.csv"))
    st.markdown("#### 🛡️ LesionNet — our model")
    st.caption("A classifier that must outline the diseased spots before it names the disease. It is trained "
               "without any hand-drawn labels, and its lesion map is the exact evidence it uses.")
    if ev is not None:
        ln = ev[ev["Model"] == "LesionNet_A3_full"]
        if len(ln):
            r = ln.iloc[0]
            c = st.columns(5)
            cards = [("Accuracy", f"{r['acc_clean']:.1%}", "best classic model: 99.8%"),
                     ("New grey background", f"{r['acc_grey_bg']:.1%}", "classic models: 45–81%"),
                     ("Evidence on disease", f"{r['e_lesion']:.0%}", "classic models: 7–12%"),
                     ("Points at a lesion", f"{r['pointing_hit']:.0%}", "classic models: 49–64%"),
                     ("Severity match ρ", f"{r['severity_spearman']:.2f}", "Grad-CAM area: ≤ 0.13")]
            for col, (label, value, ref) in zip(c, cards):
                col.metric(label, value)
                col.caption(ref)

    st.markdown("##### How it works")
    st.markdown(
        "1. **Outline:** a segmentation head marks every pixel as background, healthy leaf or diseased tissue.\n"
        "2. **Decide from the outline only:** features are averaged over the lesion pixels and the leaf pixels "
        "(*lesion-gated pooling*); background pixels never reach the classifier.\n"
        "3. **No hand labels:** training outlines come from the colour-based lesion detector; healthy-class "
        "photos are clean 'no disease' examples.\n"
        "4. **Background swap:** half of the training photos get a random new background.")

    fig_q = os.path.join(LN_FIGS, "fig_qualitative.png")
    if os.path.exists(fig_q):
        st.image(fig_q, width=820,
                 caption="Test photos. Grad-CAM++ of the classic models vs. LesionNet's lesion map "
                         "(the evidence its decision is computed from).")

    if ev is not None:
        st.markdown("##### Robustness tests (3,109 held-out photos)")
        t = ev.copy()
        t["Model"] = t["Model"].map(lambda m: NICE.get(m, m))
        show = t[["Model", "acc_clean", "macro_f1", "acc_grey_bg", "acc_swap_bg", "acc_bg_fill", "acc_bg_only"]].rename(
            columns={"acc_clean": "Accuracy", "macro_f1": "Macro-F1", "acc_grey_bg": "Grey background",
                     "acc_swap_bg": "Swapped background", "acc_bg_fill": "Background only ↓",
                     "acc_bg_only": "Silhouette only ↓"})
        st.dataframe(show.style.format({c: "{:.1%}" for c in show.columns if c not in ("Model", "Macro-F1")}
                                       | {"Macro-F1": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
        fig_v = os.path.join(LN_FIGS, "fig_variants.png")
        if os.path.exists(fig_v):
            st.image(fig_v, width="stretch", caption="The background tests on one test photo.")
        st.info("**Ablation:** *Plain EfficientNet (A0)* uses LesionNet's exact backbone, data and training, "
                "but ordinary pooling and no lesion outlines. It is just as accurate on normal photos (99.6%) "
                "but drops to 67.5% on a grey background — so the robustness comes from LesionNet's design.")

    c1, c2 = st.columns(2)
    hist = {n: os.path.join(SAVED, f"LesionNet_{t}_history.json") for n, t in (("LesionNet", "A3_full"),
                                                                                ("Plain EfficientNet (A0)", "A0_plain"))}
    with c1:
        fig = go.Figure()
        for name, p in hist.items():
            if os.path.exists(p):
                h = json.load(open(p))["history"]
                fig.add_scatter(y=h["val_cls_accuracy"], x=list(range(1, len(h["val_cls_accuracy"]) + 1)),
                                mode="lines+markers", name=name,
                                line=dict(color=theme.MODEL_COLORS["LesionNet"] if name == "LesionNet" else "#E3B35C"))
        fig.update_layout(template="agrix", height=320, title="Validation accuracy while training",
                          xaxis_title="Epoch", yaxis_tickformat=".0%", margin=dict(t=50, b=30),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if pdoc is not None:
            t = pdoc.copy()
            t["Model"] = t["Model"].map(lambda m: NICE.get(m, m))
            fig = go.Figure(go.Bar(x=t["Model"], y=t["acc12"], text=[f"{v:.0%}" for v in t["acc12"]],
                                   textposition="outside",
                                   marker_color=[theme.MODEL_COLORS.get(m.split(" ")[0], "#9AA59C") for m in t["Model"]]))
            fig.add_hline(y=1 / 12, line_dash="dash", annotation_text="chance 8%")
            fig.update_layout(template="agrix", height=320, title="Field photos (PlantDoc, never seen in training)",
                              yaxis_tickformat=".0%", yaxis_range=[0, 0.5], margin=dict(t=50, b=30))
            st.plotly_chart(fig, use_container_width=True)
    st.warning("**Open problem:** on real field photos every model, LesionNet included, scores about 20–24%. "
               "Lab photos are a poor teacher for the field; fine-tuning on field photos is the next step.")
    fig_f = os.path.join(LN_FIGS, "fig_field.png")
    if os.path.exists(fig_f):
        with st.expander("Why field photos are hard (examples)"):
            st.image(fig_f, width="stretch")

    st.markdown("##### Earlier study: do the classic models look at the disease?")
    st.caption("600 photos, Grad-CAM and Grad-CAM++ for all four classic models.")
    s1, s2 = st.columns(2)
    for col, f, cap in ((s1, "fig_attention_split.png", "Where the attention goes"),
                        (s2, "fig_camarea_vs_lesion.png", "Heatmap 'affected area' vs. real lesion area")):
        p = os.path.join(STUDY_DIR, f)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")
    p = os.path.join(STUDY_DIR, "fig_examples.png")
    if os.path.exists(p):
        with st.expander("Example heatmaps from the study"):
            st.image(p, width="stretch")


# ───────────────────────────────────────────────────────── Reports & About tab
def paper_pdf():
    p = os.path.join(PAPER_DIR, "LesionNet_paper.pdf")
    return p if os.path.exists(p) else None


def render_paper_tab():
    st.markdown(f"### {PAPER_TITLE}")
    st.markdown(f"**{PAPER_AUTHORS}**  \n{PAPER_AFFIL}")
    st.caption("Manuscript in preparation · IEEE conference format · 8 pages")
    c1, c2, _ = st.columns([1, 1, 2])
    pdf = paper_pdf()
    if pdf:
        with open(pdf, "rb") as fh:
            c1.download_button("📄 Download paper (PDF)", fh.read(), file_name="LesionNet_paper.pdf",
                               mime="application/pdf", use_container_width=True)
    src = os.path.join(PAPER_DIR, "LesionNet_paper_overleaf.zip")
    if os.path.exists(src):
        with open(src, "rb") as fh:
            c2.download_button("🧾 LaTeX source (Overleaf)", fh.read(), file_name="LesionNet_paper_overleaf.zip",
                               mime="application/zip", use_container_width=True)
    st.markdown("#### Abstract")
    st.markdown(PAPER_ABSTRACT)
    st.markdown("#### Contributions")
    st.markdown(
        "1. **Diagnostic study** — standard CNNs put 7–12% of their evidence on the disease; the most accurate "
        "one (ResNet50) is the least robust to a background change.\n"
        "2. **LesionNet** — lesion-gated pooling with a detached gate: the classifier only sees what it outlined "
        "as leaf or lesion, so its lesion map is a faithful explanation by construction.\n"
        "3. **Annotation-free training** — colour-based pseudo-masks with healthy-class negatives and "
        "reliability weighting, plus background-swap augmentation.\n"
        "4. **Robustness, faithfulness, severity** — 98.7% on a grey background (plain backbone: 67.5%), "
        "97% pointing accuracy, severity ρ = 0.90.\n"
        "5. **Honest field test** — all models fall to ≈20% on PlantDoc field photos; released openly.")
    fig_b = os.path.join(LN_FIGS, "fig_bars.png")
    if os.path.exists(fig_b):
        st.image(fig_b, width=640, caption="Accuracy under background tests and on field photos.")
    st.markdown("#### Cite")
    st.code(BIBTEX, language="bibtex")


def render_plantdoc_dataset():
    pc = _csv(os.path.join(LN_RESULTS, "plantdoc_per_class.csv"))
    st.markdown("### External test set — PlantDoc field photos")
    st.caption("Real photos from fields and the internet (Singh et al., 2020), used only for testing. "
               "12 of PlantDoc's classes match ours.")
    if pc is None:
        st.info("Run research/lesion_net/eval_external.py to add the PlantDoc results.")
        return
    c = st.columns(3)
    c[0].metric("Field photos", f"{int(pc['n'].sum()):,}")
    c[1].metric("Classes", len(pc))
    c[2].metric("Used for", "testing only")
    t = pc.copy()
    t["Class"] = t["class"].str.replace("_", " ")
    cols = [k for k in ("MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121", "LesionNet_A3_full") if k in t]
    show = t[["Class", "n"] + cols].rename(columns={"n": "Photos", "LesionNet_A3_full": "LesionNet"})
    st.markdown("**Accuracy per class on field photos**")
    st.dataframe(show.style.format({c: "{:.0%}" for c in show.columns if c not in ("Class", "Photos")}),
                 use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────── Project facts (Research Dashboard)
PV_CLASSES = [  # crop, condition, photos (PlantVillage, the 15 classes we use)
    ("Tomato", "Yellow leaf curl virus", 3208), ("Tomato", "Bacterial spot", 2127), ("Tomato", "Late blight", 1909),
    ("Tomato", "Septoria leaf spot", 1771), ("Tomato", "Spider mites", 1676), ("Tomato", "Healthy", 1591),
    ("Pepper", "Healthy", 1478), ("Tomato", "Target spot", 1404), ("Potato", "Early blight", 1000),
    ("Potato", "Late blight", 1000), ("Tomato", "Early blight", 1000), ("Pepper", "Bacterial spot", 997),
    ("Tomato", "Leaf mould", 952), ("Tomato", "Mosaic virus", 373), ("Potato", "Healthy", 152),
]
MODEL_FACTS = [
    {"Model": "MobileNetV2", "Type": "Light, fast standard CNN", "Params (M)": 2.63, "File (MB)": 27.6,
     "Accuracy": "91.6%", "Role": "classic baseline"},
    {"Model": "ResNet50", "Type": "Deep CNN with skip connections", "Params (M)": 24.78, "File (MB)": 259.2,
     "Accuracy": "99.5%", "Role": "classic baseline"},
    {"Model": "EfficientNetB0", "Type": "Balanced, efficient CNN", "Params (M)": 4.39, "File (MB)": 31.5,
     "Accuracy": "95.8%", "Role": "classic baseline"},
    {"Model": "DenseNet121", "Type": "CNN reusing all earlier features", "Params (M)": 7.34, "File (MB)": 38.1,
     "Accuracy": "96.0%", "Role": "classic baseline"},
    {"Model": "LesionNet", "Type": "EfficientNetB0 + lesion outline head + lesion-gated pooling",
     "Params (M)": 4.95, "File (MB)": 94.2, "Accuracy": "99.55%", "Role": "our model (default in app)"},
    {"Model": "LesionNet-Mobile", "Type": "Same idea on MobileNetV2", "Params (M)": 3.35, "File (MB)": 63.7,
     "Accuracy": "95.98%", "Role": "our light model"},
    {"Model": "Plain EfficientNet (A0)", "Type": "LesionNet's backbone trained the ordinary way",
     "Params (M)": 4.88, "File (MB)": None, "Accuracy": "99.6%", "Role": "ablation only"},
]


def render_project_facts():
    st.markdown("#### 📘 Project at a glance")
    st.caption("The basic facts: data, splits, models, parameters and training settings.")
    c = st.columns(5)
    c[0].metric("Training photos", "20,638")
    c[1].metric("Classes", "15")
    c[2].metric("Crops", "3")
    c[3].metric("Field test photos", "1,012")
    c[4].metric("Models trained", "7")

    st.markdown("##### Dataset split (PlantVillage, stratified per class, seed 42)")
    s1, s2 = st.columns([1, 1.3])
    split = pd.DataFrame({"Split": ["Train", "Validation", "Test"], "Share": ["70%", "15%", "15%"],
                          "Photos": [14440, 3089, 3109]})
    s1.dataframe(split.assign(Photos=split["Photos"].map("{:,}".format)), use_container_width=True, hide_index=True)
    s1.caption("7 models = 4 classic + LesionNet + LesionNet-Mobile + 1 ablation (A0).")
    fig = go.Figure(go.Pie(labels=split["Split"], values=split["Photos"], hole=0.55, sort=False,
                           marker=dict(colors=[theme.GREEN, "#9AA59C", theme.MODEL_COLORS["LesionNet"]]),
                           textinfo="label+value"))
    fig.update_layout(template="agrix", height=240, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    s2.plotly_chart(fig, use_container_width=True)
    st.caption("Images resized to 224 × 224. Classes are imbalanced (3,208 vs 152 photos, about 21×), "
               "handled with class weights (0.5–3×) in the loss.")

    st.markdown("##### Classes and photo counts")
    cls = pd.DataFrame(PV_CLASSES, columns=["Crop", "Condition", "Photos"])
    fig = go.Figure(go.Bar(x=cls["Photos"], y=cls["Crop"] + " · " + cls["Condition"], orientation="h",
                           marker_color=[{"Tomato": "#C2410C", "Potato": theme.PURPLE, "Pepper": theme.GREEN}[c]
                                         for c in cls["Crop"]],
                           text=[f"{v:,}" for v in cls["Photos"]], textposition="outside"))
    fig.update_layout(template="agrix", height=470, yaxis=dict(autorange="reversed"),
                      margin=dict(t=10, b=10, l=10, r=40), xaxis_title="Photos")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Second dataset: PlantDoc (testing only)")
    st.markdown("Real field and internet photos (Singh et al., 2020): **1,012 photos, 12 of our classes** "
                "(no healthy potato, target spot or spider mites). Never used for training, only to test "
                "how models cope with messy real-world photos.")

    st.markdown("##### Automatic training outlines (for LesionNet)")
    st.markdown("- **17,895 of 20,638** photos (86.7%) got a usable outline from our colour-based detector "
                "(background / healthy leaf / lesion).\n"
                "- Lesions cover on average **12.8%** of the leaf (median 8.8%); the leaf covers **42.5%** of the photo.\n"
                "- No outline was drawn by hand.")

    st.markdown("##### Models and parameters")
    mf = pd.DataFrame(MODEL_FACTS)
    st.dataframe(mf.astype(object).where(mf.notna(), "—"), use_container_width=True, hide_index=True)
    st.caption("Classic models: accuracy on their original test split. LesionNet models: on the 3,109-photo "
               "held-out split (where the classic models score 93.0%, 99.8%, 96.4% and 96.2%).")

    st.markdown("##### Training settings")
    t1, t2 = st.columns(2)
    t1.markdown("**Classic models (transfer learning)**\n\n"
                "| Setting | Value |\n|---|---|\n"
                "| Pretrained on | ImageNet |\n| Input size | 224 × 224 |\n| Batch size | 32 |\n"
                "| Phase 1 | 5 epochs, backbone frozen, lr 0.0001 |\n"
                "| Phase 2 | 15 epochs, last 50 layers fine-tuned, lr 0.00001 |\n"
                "| Head | GAP → BatchNorm → Dense → Dropout 0.3 → Softmax (15) |")
    t2.markdown("**LesionNet**\n\n"
                "| Setting | Value |\n|---|---|\n"
                "| Optimizer | Adam |\n| Phase 1 | 3 epochs, backbone frozen, lr 0.001 |\n"
                "| Phase 2 | 15 epochs, whole network, cosine decay 0.0001 → 0.000001 |\n"
                "| Batch size | 32, mixed precision (16-bit) |\n"
                "| Loss | classification + 1 × outline loss (lesion pixels ×3) |\n"
                "| Augmentation | flips, rotations, colour jitter, background swap (50%) |\n"
                "| Outline map / features | 56 × 56 / 14 × 14 × 256 |\n"
                "| Early stopping | patience 5, best validation epoch kept |")
    st.markdown("##### Hardware and time")
    st.markdown("NVIDIA RTX 4050 Laptop GPU (6 GB), Ubuntu on WSL2, TensorFlow / Keras 3. "
                "LesionNet trained in **76 minutes**, the plain baseline (A0) in 42 minutes. "
                "LesionNet-Mobile stopped at epoch 10 of 18 after a GPU driver crash (best epoch used).")
