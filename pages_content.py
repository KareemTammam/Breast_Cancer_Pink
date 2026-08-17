import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style import COLORS, DARK_OVERRIDES, MODEL_PALETTE, METRIC_COLORS, page_header
from utils import (
    build_raw_input_row,
    load_confusion_matrices,
    load_feature_importance,
    load_feature_names,
    load_feature_stats,
    load_meta,
    load_model,
    load_model_comparison,
    load_sample_data,
    load_scaler,
    load_target_correlation,
    prettify_feature,
)


def _is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def chart_colors() -> dict:
    """Text/gridline colors for the CURRENT theme, so charts always match
    the page instead of staying frozen on the light palette."""
    if _is_dark():
        return {"text": DARK_OVERRIDES["text"], "grid": DARK_OVERRIDES["border"]}
    return {"text": COLORS["text"], "grid": "#F1F5F9"}


def get_plotly_layout(margin=None) -> dict:
    """Base layout for every chart. Backgrounds are fully transparent
    (not white/dark-colored) so the figure sits directly on the page
    background instead of showing up as its own box in dark mode."""
    cc = chart_colors()
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=cc["text"]),
        margin=margin or dict(l=10, r=10, t=40, b=10),
    )

# Feature groups, in the same order train.py selected them
MEAN_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean",
]
SE_FEATURES = [
    "radius_se", "perimeter_se", "area_se", "compactness_se",
    "concavity_se", "concave points_se",
]
WORST_FEATURES = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HOME
# =============================================================================
def render_home():
    meta = load_meta()
    comparison = load_model_comparison()
    best = comparison.iloc[0]

    left, right = st.columns([1.3, 1])
    with left:
        # All hero content must be emitted in a SINGLE st.markdown call —
        # Streamlit renders each st.markdown as its own isolated DOM
        # container, so a <div> opened in one call and closed in another
        # does NOT wrap the elements in between (the browser auto-closes
        # it), leaving an empty colored box floating above the text.
        st.markdown(
            """
            <div class="hero-frame">
                <div class="badge badge-blue">Machine Learning · Binary Classification</div>
                <div class="hero-title">🎗️ Breast Cancer Diagnosis Prediction</div>
                <div class="hero-sub">Predicts whether a breast mass is malignant or benign
                from digitized fine needle aspirate (FNA) cell nuclei measurements —
                trained on the Wisconsin Diagnostic Breast Cancer dataset and benchmarked
                across 8 ML models.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🎯  Make a Prediction", width='stretch'):
                st.session_state.nav_request = "Make a Prediction"
                st.rerun()
        with c2:
            if st.button("📊  View Model Dashboard", width='stretch'):
                st.session_state.nav_request = "Model Dashboard"
                st.rerun()

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div class="kpi-label">Best Model</div>
                <div class="kpi-value" style="font-size:1.6rem;">{best['Model']}</div>
                <hr style="margin:0.9rem 0;">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div class="kpi-label">F1 Score</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:{COLORS['primary']};">{best['F1']*100:.1f}%</div>
                    </div>
                    <div>
                        <div class="kpi-label">Accuracy</div>
                        <div class="kpi-value" style="font-size:1.4rem; color:{COLORS['success']};">{best['Accuracy']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    st.markdown("#### What this project does")
    st.markdown(
        "<p class='muted'>Cell nuclei from a digitized image of a breast mass FNA are "
        "described by 30 measurements — radius, texture, perimeter, area, smoothness, "
        "compactness, concavity, and more, each summarized as a mean, standard error, "
        "and 'worst' (largest) value. This tool learns from 569 diagnosed cases to "
        "predict whether a new mass is <b>malignant</b> or <b>benign</b>, helping "
        "prioritize which cases need closer clinical review.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    cols = st.columns(4)
    stats = [
        ("Training Samples", f"{meta['n_samples']:,}", "Diagnosed FNA cases"),
        ("Selected Features", f"{meta['n_features_encoded']}", "Correlation-filtered from 30"),
        ("Models Benchmarked", "8", "Classical ML to gradient boosting"),
        ("Best F1 Score", f"{best['F1']*100:.1f}%", f"Achieved by {best['Model']}"),
    ]
    for col, (label, val, sub) in zip(cols, stats):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    st.write("")
    st.markdown("#### How it works")
    steps = [
        ("Enter cell nuclei measurements", "Radius, texture, area, concavity, and other FNA-derived values."),
        ("Model processes the input", "The same scaling pipeline used in training is applied instantly."),
        ("Get an instant prediction", "See the likely diagnosis with a confidence score, in a clear result card."),
    ]
    scols = st.columns(3)
    for i, (col, (title, desc)) in enumerate(zip(scols, steps)):
        with col:
            st.markdown(
                f"""
                <div class="card" style="min-height:130px;">
                    <span class="step-num">{i+1}</span><b>{title}</b>
                    <p class="muted" style="margin-top:0.5rem; font-size:0.9rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.info(
        "⚕️ **Not a diagnostic tool.** This app is a machine-learning demo for "
        "educational purposes and should never replace evaluation by a qualified "
        "medical professional.",
        icon="⚕️",
    )


# =============================================================================
# DATASET
# =============================================================================
def render_dataset():
    meta = load_meta()
    sample = load_sample_data()
    corr = load_target_correlation()

    page_header(
        "Dataset Overview",
        "Features computed from a digitized image of a fine needle aspirate (FNA) of "
        "a breast mass, describing characteristics of the cell nuclei present in the "
        "image. The classification target is whether the mass is malignant or benign.",
        icon="🗂️",
    )

    cols = st.columns(4)
    class_counts = meta["class_counts"]
    total = sum(class_counts.values())
    stats = [
        ("Total Samples", f"{meta['n_samples']:,}", ""),
        ("Selected Features", f"{meta['n_features_raw']}", "Correlation-filtered from 30"),
        ("Train / Test Split", f"{meta['n_train']:,} / {meta['n_test']:,}", "80 / 20 stratified"),
        ("Class Balance", f"{class_counts.get('1', class_counts.get(1, 0))/total*100:.1f}% Malignant",
         f"{class_counts.get('0', class_counts.get(0, 0))/total*100:.1f}% Benign"),
    ]
    for col, (label, val, sub) in zip(cols, stats):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("##### Target Distribution")
        counts = {"Benign": class_counts.get("0", class_counts.get(0, 0)),
                  "Malignant": class_counts.get("1", class_counts.get(1, 0))}
        fig = go.Figure(
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.62,
                marker=dict(colors=[COLORS["success"], COLORS["danger"]]),
                textinfo="label+percent",
                textfont=dict(size=13, family="Inter"),
            )
        )
        fig.update_layout(
            **get_plotly_layout(),
            height=320,
            showlegend=False,
            annotations=[dict(text="Diagnosis", x=0.5, y=0.5, font_size=16, showarrow=False,
                               font=dict(color=chart_colors()["text"], family="Inter"))],
        )
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("##### Top Features Correlated with Diagnosis")
        top_corr = corr.head(10).sort_values()
        bar_colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in top_corr.values]
        fig2 = go.Figure(
            go.Bar(
                x=top_corr.values,
                y=[prettify_feature(f) for f in top_corr.index],
                orientation="h",
                marker_color=bar_colors,
            )
        )
        fig2.update_layout(
            **get_plotly_layout(),
            height=320,
            xaxis_title="Correlation with target (malignant = 1)",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig2, width='stretch')

    st.write("")
    st.markdown("##### Sample of the Data")
    st.dataframe(sample.head(12), width='stretch', hide_index=True)

    st.write("")
    st.markdown("##### Feature Reference")
    st.markdown(
        "<p class='muted' style='font-size:0.9rem;'>Ten base measurements are taken "
        "for each cell nucleus, each summarized three ways: the <b>mean</b>, the "
        "<b>standard error</b>, and the <b>worst</b> (average of the three largest "
        "values). 25 of the 30 resulting columns passed the correlation-with-target "
        "threshold and were used for modeling.</p>",
        unsafe_allow_html=True,
    )
    feat_info = pd.DataFrame(
        [
            ("radius", "Mean of distances from center to points on the perimeter"),
            ("texture", "Standard deviation of gray-scale values"),
            ("perimeter", "Perimeter of the nucleus"),
            ("area", "Area of the nucleus"),
            ("smoothness", "Local variation in radius lengths"),
            ("compactness", "perimeter² / area − 1.0"),
            ("concavity", "Severity of concave portions of the contour"),
            ("concave points", "Number of concave portions of the contour"),
            ("symmetry", "Symmetry of the nucleus"),
            ("fractal dimension", "'Coastline approximation' − 1"),
        ],
        columns=["Base Measurement", "Description"],
    )
    st.dataframe(feat_info, width='stretch', hide_index=True)


# =============================================================================
# MODEL DASHBOARD
# =============================================================================
def render_dashboard():
    comparison = load_model_comparison()
    fi = load_feature_importance()
    cms = load_confusion_matrices()
    best = comparison.iloc[0]

    page_header(
        "Model Performance Dashboard",
        "8 classification models were trained and tuned via grid search cross-validation, "
        "then evaluated on a held-out test set.",
        icon="📊",
    )

    cols = st.columns(4)
    kpis = [
        ("Best Model", best["Model"], "Highest F1 score"),
        ("Accuracy", f"{best['Accuracy']*100:.1f}%", ""),
        ("Precision", f"{best['Precision']*100:.1f}%", ""),
        ("Recall", f"{best['Recall']*100:.1f}%", ""),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        with col:
            kpi_card(label, val, sub)

    st.write("")
    st.markdown("##### Model Comparison — Accuracy, Precision, Recall, F1")
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=comparison["Model"],
                y=comparison[metric],
                marker_color=METRIC_COLORS[metric],
                text=[f"{v:.2f}" for v in comparison[metric]],
                textposition="outside",
                textfont=dict(size=10),
            )
        )
    fig.update_layout(
        **get_plotly_layout(),
        barmode="group",
        height=460,
        yaxis=dict(range=[0, 1.08], title="Score", gridcolor=chart_colors()["grid"]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        bargap=0.25,
        bargroupgap=0.08,
    )
    st.plotly_chart(fig, width='stretch')

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"##### 🧮 Confusion Matrix — {best['Model']}")
        model_names = list(cms.keys())
        default_idx = model_names.index(best["Model"]) if best["Model"] in model_names else 0
        chosen = st.selectbox("Select a model", model_names, index=default_idx, key="cm_select")
        cm = np.array(cms[chosen])
        labels = ["Benign", "Malignant"]
        cm_max = cm.max() if cm.max() > 0 else 1
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#FBCFE8"], [1, COLORS["primary"]]],
                zmin=0,
                zmax=cm_max,
                showscale=False,
            )
        )
        # Plotly's Heatmap textfont.color only accepts a single color, not
        # a per-cell array, so per-cell readable text (white on the dark
        # cells, dark navy on the light cells) is added as annotations
        # instead of texttemplate.
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                value = int(cm[i][j])
                text_color = "#FFFFFF" if value >= cm_max * 0.45 else chart_colors()["text"]
                fig_cm.add_annotation(
                    x=col_label, y=row_label, text=str(value),
                    showarrow=False, font=dict(size=18, color=text_color),
                )
        fig_cm.update_layout(
            **get_plotly_layout(),
            height=340,
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="Actual", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width='stretch')

    with right:
        st.markdown(f"##### Top Features — {best['Model']}")
        fi_sorted = fi.sort_values("importance").tail(10)
        fig_fi = go.Figure(
            go.Bar(
                x=fi_sorted["importance"],
                y=[prettify_feature(f) for f in fi_sorted["feature"]],
                orientation="h",
                marker_color=COLORS["primary"],
            )
        )
        fig_fi.update_layout(
            **get_plotly_layout(),
            height=340,
            xaxis_title="Importance",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig_fi, width='stretch')

    st.write("")
    st.markdown("##### Full Results Table")
    display_df = comparison.copy()
    for m in metrics:
        display_df[m] = (display_df[m] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, width='stretch', hide_index=True)


# =============================================================================
# PREDICTION
# =============================================================================
def _num_input(label, col, stats, key):
    lo, hi, mean = stats["min"], stats["max"], stats["mean"]
    span = hi - lo
    step = round(max(span / 100, 0.0001), 4)
    pad = span * 0.2
    return st.number_input(
        label,
        min_value=round(max(0.0, lo - pad), 4),
        max_value=round(hi + pad, 4),
        value=round(mean, 4),
        step=step,
        format="%.4f",
        key=key,
    )


def render_predict():
    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    stats = load_feature_stats()

    page_header(
        "Make a Prediction",
        "Enter the cell nuclei measurements below. The exact scaling pipeline used "
        "during training is applied automatically.",
        icon="🔮",
    )
    st.write("")

    with st.form("prediction_form"):
        st.markdown("##### 📏 Mean Values")
        st.caption("Average of the measurement across all cell nuclei in the image.")
        form_values = {}
        cols = st.columns(3)
        for i, feat in enumerate([f for f in MEAN_FEATURES if f in feature_names]):
            with cols[i % 3]:
                form_values[feat] = _num_input(prettify_feature(feat), feat, stats[feat], key=f"in_{feat}")

        st.write("")
        st.markdown("##### 📐 Standard Error Values")
        st.caption("Variability of the measurement across the nuclei in the image.")
        cols = st.columns(3)
        for i, feat in enumerate([f for f in SE_FEATURES if f in feature_names]):
            with cols[i % 3]:
                form_values[feat] = _num_input(prettify_feature(feat), feat, stats[feat], key=f"in_{feat}")

        st.write("")
        st.markdown("##### 📈 Worst Values")
        st.caption("Average of the three largest values of the measurement in the image.")
        cols = st.columns(3)
        for i, feat in enumerate([f for f in WORST_FEATURES if f in feature_names]):
            with cols[i % 3]:
                form_values[feat] = _num_input(prettify_feature(feat), feat, stats[feat], key=f"in_{feat}")

        st.write("")
        submitted = st.form_submit_button("🔮  Predict Diagnosis", width='stretch')

    if submitted:
        with st.spinner("Running the model..."):
            raw_row = build_raw_input_row(form_values, feature_names)
            try:
                scaled = scaler.transform(raw_row)
                proba = model.predict_proba(scaled)[0]
                pred = int(np.argmax(proba))
                confidence = proba[pred]
            except Exception as e:
                st.error(f"Something went wrong while running the prediction: {e}")
                return

        st.write("")
        result_class = "result-no" if pred == 1 else "result-yes"
        result_text = "MALIGNANT ⚠️" if pred == 1 else "BENIGN ✅"
        explanation = (
            "Based on these cell nuclei measurements, the model finds a strong signal "
            "consistent with malignant tumors. This should be followed up by a "
            "qualified medical professional."
            if pred == 1 else
            "Based on these cell nuclei measurements, the model finds this profile "
            "resembles benign masses in the training data."
        )

        rc1, rc2 = st.columns([1.4, 1])
        with rc1:
            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value">{result_text}</div>
                    <div class="muted" style="font-size:0.92rem;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc2:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%", "font": {"size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": COLORS["danger"] if pred == 1 else COLORS["success"]},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": chart_colors()["grid"],
                        "borderwidth": 1,
                    },
                    title={"text": "Confidence Score", "font": {"size": 14}},
                )
            )
            fig.update_layout(**get_plotly_layout(margin=dict(l=20, r=20, t=40, b=10)), height=230)
            st.plotly_chart(fig, width='stretch')

        st.write("")
        st.markdown("##### Prediction Breakdown")
        bd1, bd2 = st.columns([1, 1.3])

        with bd1:
            fig_proba = go.Figure(
                go.Bar(
                    x=["benign", "malignant"],
                    y=[float(proba[0]), float(proba[1])],
                    marker_color=[COLORS["success"], COLORS["danger"]],
                    text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    textposition="outside",
                    textfont=dict(size=14, family="Inter"),
                    width=[0.55, 0.55],
                )
            )
            fig_proba.update_layout(
                **get_plotly_layout(),
                height=300,
                yaxis=dict(title="probability", range=[0, 1.15], gridcolor=chart_colors()["grid"]),
                xaxis=dict(title="class"),
            )
            st.plotly_chart(fig_proba, width='stretch')

        with bd2:
            fi = load_feature_importance()
            fi_sorted = fi.sort_values("importance").tail(8)
            fig_drivers = go.Figure(
                go.Bar(
                    x=fi_sorted["importance"],
                    y=[prettify_feature(f) for f in fi_sorted["feature"]],
                    orientation="h",
                    marker=dict(
                        color=fi_sorted["importance"],
                        colorscale=[
                            [0, "#FDE68A"], [0.5, "#F97316"], [1, "#7C3AED"],
                        ],
                        showscale=True,
                        colorbar=dict(title="importance", thickness=12, len=0.8),
                    ),
                )
            )
            fig_drivers.update_layout(
                **get_plotly_layout(),
                height=300,
                title=dict(text="Overall top drivers of the model (global importance)", font=dict(size=13)),
                xaxis_title="importance",
                yaxis=dict(automargin=True),
            )
            st.plotly_chart(fig_drivers, width='stretch')

        st.write("")
        st.markdown("##### Values You Entered")
        display_row = raw_row.copy()
        display_row.columns = [prettify_feature(c) for c in display_row.columns]
        st.dataframe(display_row, width='stretch', hide_index=True)

        best_model_name = load_meta().get("best_model_name", type(model).__name__)
        st.caption(
            f"Built with Streamlit • {best_model_name} pre-trained on the Wisconsin "
            "Breast Cancer dataset • model & scaler loaded from model/trained_model.pkl"
        )
        st.warning(
            "⚕️ This prediction is for educational purposes only and is not a "
            "medical diagnosis. Always consult a qualified healthcare professional.",
            icon="⚕️",
        )


# =============================================================================
# ABOUT
# =============================================================================
def render_about():
    comparison = load_model_comparison()
    best = comparison.iloc[0]

    page_header("About This Project", icon="ℹ️")

    st.markdown("##### 🎯 Problem")
    st.markdown(
        "<p class='muted'>Fine needle aspiration (FNA) is a minimally invasive biopsy "
        "technique used to sample a breast mass. Once digitized, an image of the "
        "sample yields dozens of measurements describing the cell nuclei present. "
        "This project predicts — from those measurements alone — whether a mass is "
        "malignant or benign, to support faster, more consistent triage.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 🗂️ Dataset")
    st.markdown(
        "<p class='muted'>569 diagnosed cases (the Wisconsin Diagnostic Breast Cancer "
        "dataset), each with 30 numeric features describing cell nuclei from a digitized "
        "FNA image — 10 base measurements (radius, texture, perimeter, area, smoothness, "
        "compactness, concavity, concave points, symmetry, fractal dimension), each given "
        "as a mean, standard error, and worst value. The target variable "
        "<b>diagnosis</b> indicates malignant (M) or benign (B).</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 🛠️ Preprocessing")
    steps = [
        "Dropped the `id` column since it carries no predictive signal.",
        "Encoded the target `diagnosis` to 0 (benign) / 1 (malignant) with `LabelEncoder`.",
        "Kept only features with an absolute correlation to the target above 0.20, "
        "reducing 30 raw columns to 25.",
        "Stratified 80/20 train-test split, `random_state=42`.",
        "All 25 selected features standardized with `StandardScaler`, fit on the "
        "training set only.",
    ]
    for s in steps:
        st.markdown(f"<div style='margin-bottom:0.4rem;'>• <span class='muted'>{s}</span></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("##### 🤖 Models Tested")
    model_list = ", ".join(comparison["Model"].tolist())
    st.markdown(
        f"<p class='muted'>Eight classifiers were trained, with hyperparameters "
        f"tuned via grid search cross-validation where applicable: {model_list}.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 📏 Evaluation Metrics")
    metric_cards = st.columns(4)
    metric_defs = [
        ("Accuracy", "Overall share of correct predictions."),
        ("Precision", "Of predicted malignant cases, how many actually were."),
        ("Recall", "Of actual malignant cases, how many were correctly identified."),
        ("F1 Score", "Harmonic mean of precision and recall — the primary model-selection metric."),
    ]
    for col, (name, desc) in zip(metric_cards, metric_defs):
        with col:
            st.markdown(
                f"""<div class="card" style="min-height:130px;">
                <b>{name}</b>
                <p class="muted" style="font-size:0.85rem; margin-top:0.4rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("##### 🏆 Best Model")
    st.markdown(
        f"""
        <div class="card">
        <span class="badge badge-green">Selected</span>
        <h4 style="margin-top:0.6rem;">{best['Model']}</h4>
        <p class="muted">
        {best['Model']} achieved the highest F1 score ({best['F1']*100:.1f}%) among all
        candidates, with {best['Accuracy']*100:.1f}% accuracy, {best['Precision']*100:.1f}%
        precision, and {best['Recall']*100:.1f}% recall on the held-out test set. With a tuned
        RBF kernel, it separates the malignant and benign classes cleanly even though the
        classes aren't linearly separable in the raw feature space — the standardized,
        correlation-filtered inputs give it a clean signal to work with.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("##### 🔗 Tech Stack")
    st.markdown(
        "<p class='muted'>Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.info(
        "⚕️ This project is for educational/portfolio purposes only and is not "
        "validated for clinical use.",
        icon="⚕️",
    )
