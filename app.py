
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Women Safety Analytics",
    page_icon="🛡️",
    layout="wide"
)

root = Path(__file__).resolve().parent
data_path = root / "data" / "processed"

st.markdown("""
<style>
.main {
    background-color: #faf8fc;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #713b69;
}
div[data-testid="stMetric"] {
    background-color: #f1e8f3;
    border: 1px solid #ead9ed;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(filename):
    path = data_path / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def find_column(df, candidates):
    if df is None:
        return None

    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def show_missing_file(filename):
    st.warning(
        f"{filename} was not found in data/processed. "
        "Run the relevant ML script to generate it."
    )


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


report = load_data("final_state_clustering_report.csv")
features = load_data("state_features_2024.csv")
models = load_data("model_comparison_results.csv")
cluster_summary = load_data("final_cluster_summary.csv")
anomaly_data = load_data("state_anomaly_results.csv")
predictions = load_data("model_comparison_predictions.csv")

if report is None:
    st.error(
        "The file final_state_clustering_report.csv is missing. "
        "Please run ML17_final_clustering_report.py first."
    )
    st.stop()

state_col = find_column(
    report,
    ["State_UT", "State", "State/UT", "State_Name", "State_UT_Name"]
)

cluster_col = find_column(report, ["Cluster"])
anomaly_col = find_column(report, ["Anomaly", "Anomaly_Label"])
score_col = find_column(
    report,
    ["Anomaly_Score", "Anomaly Score", "Score"]
)

if state_col is None:
    st.error(
        "Could not identify the state column in "
        "final_state_clustering_report.csv."
    )
    st.write("Available columns:", report.columns.tolist())
    st.stop()

report[state_col] = report[state_col].astype(str).str.strip()
report = report[
    ~report[state_col].str.upper().isin(["TOTAL", "ALL INDIA", "INDIA"])
].copy()

states = sorted(report[state_col].dropna().unique().tolist())

if not states:
    st.error("No state or union territory records were found.")
    st.stop()

st.sidebar.title("🛡️ Women Safety Analytics")
st.sidebar.caption("Machine Learning • India")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "State Explorer",
        "Cluster Analysis",
        "Anomaly Detection",
        "Model Performance",
        "Data Explorer",
        "Women's Safety Awareness"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "This application analyses historical reported crime patterns. "
    "It is not a real-time safety or emergency service."
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("🛡️ Women Safety Analytics")
    st.subheader("Understanding Women's Safety Through Data")

    st.markdown("""
    This project uses historical reported crime data and Machine
    Learning to understand crime patterns across Indian states
    and union territories.

    It helps us compare patterns, group similar states, identify
    unusual observations and evaluate models that estimate crime rates.
    """)

    total_states = len(report)

    total_clusters = (
        report[cluster_col].nunique()
        if cluster_col else 0
    )

    if anomaly_col:
        labels = pd.to_numeric(report[anomaly_col], errors="coerce")
        anomalies = int((labels == -1).sum())
        normal = int((labels == 1).sum())
    else:
        anomalies = 0
        normal = 0

    best_model = "Not available"
    if models is not None:
        model_col = find_column(models, ["Model", "Algorithm"])
        mae_col = find_column(models, ["MAE"])

        if model_col and mae_col:
            valid_models = models.dropna(subset=[mae_col])
            if not valid_models.empty:
                best_model = valid_models.loc[
                    valid_models[mae_col].idxmin(), model_col
                ]

    st.header("Project at a Glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States / UTs", total_states)
    c2.metric("Clusters", total_clusters)
    c3.metric("Normal Patterns", normal)
    c4.metric("Anomalies", anomalies)

    st.divider()

    st.header("What is this project trying to achieve?")

    a, b = st.columns(2)

    with a:
        st.subheader("Understanding crime patterns")
        st.write("""
        We analyse reported crimes against women across states
        and union territories to understand how different crime
        categories are distributed.
        """)

        st.subheader("Finding similarities")
        st.write("""
        K-Means groups states that show similar patterns across
        the selected crime-rate features.
        """)

    with b:
        st.subheader("Finding unusual patterns")
        st.write("""
        Isolation Forest identifies observations that differ
        statistically from the majority of the dataset.
        """)

        st.subheader("Estimating crime rates")
        st.write("""
        Regression models are compared to understand how well
        they estimate numerical crime-rate values.
        """)

    st.divider()

    st.header("Best Performing Regression Model")

    if best_model != "Not available":
        st.success(f"Best model based on MAE: {best_model}")
        st.caption(
            "The model with the lowest Mean Absolute Error is selected. "
            "This is a comparison on the available evaluation data, "
            "not a guarantee of future prediction accuracy."
        )
    else:
        show_missing_file("model_comparison_results.csv")

    st.divider()

    st.header("Cluster Distribution")

    if cluster_col:
        cluster_counts = (
            report[cluster_col]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        cluster_counts.columns = ["Cluster", "Number of States"]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Number of States",
            color="Cluster",
            text="Number of States",
            title="Number of States in Each Cluster"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Each cluster represents a group of states with similar "
            "crime-rate patterns. Cluster numbers are labels, not "
            "safety rankings."
        )

    st.divider()

    st.header("Anomaly Overview")

    if anomaly_col:
        anomaly_counts = pd.Series({
            "Normal": normal,
            "Anomaly": anomalies
        }).reset_index()
        anomaly_counts.columns = ["Pattern", "Count"]

        fig = px.pie(
            anomaly_counts,
            names="Pattern",
            values="Count",
            hole=0.5,
            title="Normal and Unusual Observations"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "An anomaly is a statistically unusual observation. "
            "It does not automatically indicate that a state is unsafe."
        )


# --------------------------------------------------
# STATE EXPLORER
# --------------------------------------------------

elif page == "State Explorer":

    st.title("📍 State-wise Explorer")
    st.write(
        "Select a state or union territory to explore its available "
        "crime-pattern and Machine Learning results."
    )

    selected_state = st.selectbox(
        "Choose a State / Union Territory",
        states
    )

    state_row = report[
        report[state_col] == selected_state
    ].iloc[0]

    st.subheader(selected_state)

    c1, c2, c3 = st.columns(3)

    if cluster_col:
        c1.metric("Cluster", str(state_row[cluster_col]))

    if anomaly_col:
        label = pd.to_numeric(
            pd.Series([state_row[anomaly_col]]),
            errors="coerce"
        ).iloc[0]

        status = (
            "Anomaly detected"
            if label == -1
            else "Normal pattern"
            if label == 1
            else "Not available"
        )
        c2.metric("Pattern Status", status)

    if score_col:
        c3.metric(
            "Anomaly Score",
            f"{float(state_row[score_col]):.4f}"
        )

    st.divider()

    st.subheader("State Information")

    st.dataframe(
        state_row.to_frame("Value"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Crime Rate Features")

    state_features = None

    if features is not None:
        feature_state_col = find_column(
            features,
            ["State_UT", "State", "State_Name"]
        )

        if feature_state_col:
            match = features[
                features[feature_state_col].astype(str).str.strip()
                == selected_state
            ]

            if not match.empty:
                state_features = match.iloc[0]

    if state_features is not None:
        rate_cols = [
            c for c in numeric_columns(features)
            if "rate" in c.lower()
        ]

        if rate_cols:
            chart_data = pd.DataFrame({
                "Crime Category": rate_cols,
                "Rate": [
                    pd.to_numeric(
                        pd.Series([state_features[c]]),
                        errors="coerce"
                    ).iloc[0]
                    for c in rate_cols
                ]
            }).dropna()

            if not chart_data.empty:
                fig = px.bar(
                    chart_data,
                    x="Crime Category",
                    y="Rate",
                    color="Rate",
                    title=f"Crime Rate Features: {selected_state}"
                )
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric rate values are available.")
        else:
            st.info("No rate columns were found in the feature dataset.")
    else:
        st.info(
            "State-level feature data is unavailable or its state "
            "column could not be identified."
        )

    st.caption(
        "These figures describe historical reported crime patterns, "
        "not the real-time safety of this state or its individual locations."
    )


# --------------------------------------------------
# CLUSTER ANALYSIS
# --------------------------------------------------

elif page == "Cluster Analysis":

    st.title("🔍 Cluster Analysis")
    st.write("""
    K-Means clustering groups states and union territories
    based on similarities in their selected crime-rate features.
    """)

    if not cluster_col:
        st.warning("Cluster column is missing from the report.")
    else:
        cluster_values = sorted(
            report[cluster_col].dropna().unique().tolist()
        )

        selected_cluster = st.selectbox(
            "Select a Cluster",
            cluster_values
        )

        cluster_states = report[
            report[cluster_col] == selected_cluster
        ]

        st.metric("States in Selected Cluster", len(cluster_states))

        st.subheader("States in this Cluster")

        st.dataframe(
            cluster_states,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Cluster Size Comparison")

        counts = (
            report.groupby(cluster_col)[state_col]
            .count()
            .reset_index(name="Number of States")
        )

        fig = px.bar(
            counts,
            x=cluster_col,
            y="Number of States",
            color=cluster_col,
            text="Number of States",
            title="State Distribution Across Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        A cluster is simply a group of similar observations.
        A state in Cluster 0 is not automatically safer or less safe
        than a state in Cluster 3.
        """)


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

elif page == "Anomaly Detection":

    st.title("⚠️ Anomaly Detection")

    st.write("""
    Isolation Forest identifies statistically unusual observations
    based on their selected crime-rate patterns.
    """)

    if anomaly_col is None:
        st.warning("Anomaly column is unavailable.")
    else:
        labels = pd.to_numeric(
            report[anomaly_col],
            errors="coerce"
        )

        filter_choice = st.selectbox(
            "Filter observations",
            ["All", "Normal", "Anomalies"]
        )

        if filter_choice == "Normal":
            view = report[labels == 1]
        elif filter_choice == "Anomalies":
            view = report[labels == -1]
        else:
            view = report.copy()

        st.metric("Observations Displayed", len(view))

        if score_col:
            sort_order = st.checkbox(
                "Show most unusual observations first",
                value=True
            )
            view = view.sort_values(
                score_col,
                ascending=sort_order
            )

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True
        )

        st.info("""
        Anomaly detection highlights unusual statistical patterns.
        It does not prove that a state has more crime or is unsafe.
        Further investigation and reliable additional data are needed.
        """)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write("""
    This section compares the regression models used to estimate
    crime rates.
    """)

    if models is None:
        show_missing_file("model_comparison_results.csv")
    else:
        st.subheader("Model Comparison")

        st.dataframe(
            models,
            use_container_width=True,
            hide_index=True
        )

        model_col = find_column(models, ["Model", "Algorithm"])

        metric_cols = [
            c for c in ["MAE", "RMSE", "R2", "MAPE"]
            if c in models.columns
        ]

        if model_col and metric_cols:
            selected_metric = st.selectbox(
                "Choose evaluation metric",
                metric_cols
            )

            chart_data = models.copy()
            chart_data[selected_metric] = pd.to_numeric(
                chart_data[selected_metric],
                errors="coerce"
            )

            fig = px.bar(
                chart_data,
                x=model_col,
                y=selected_metric,
                color=model_col,
                text=selected_metric,
                title=f"Model Comparison Using {selected_metric}"
            )

            st.plotly_chart(fig, use_container_width=True)

            if selected_metric in ["MAE", "RMSE", "MAPE"]:
                st.info(
                    f"Lower {selected_metric} generally indicates "
                    "smaller prediction errors."
                )
            else:
                st.info(
                    "A higher R² generally indicates a better fit "
                    "on the evaluated data."
                )

        if predictions is not None:
            st.divider()
            st.subheader("Actual vs Predicted Values")

            st.write(
                "Explore the available prediction records below."
            )

            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Prediction-level data is unavailable. "
                "Run the model comparison script to generate it."
            )


# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------

elif page == "Data Explorer":

    st.title("📊 Data Explorer")

    st.write("""
    Search, filter, and download the state-wise Machine Learning report.
    """)

    search = st.text_input(
        "Search for a State / Union Territory"
    )

    view = report.copy()

    if search:
        view = view[
            view[state_col].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.write(f"Records found: {len(view)}")

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True
    )

    csv = view.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Report",
        data=csv,
        file_name="women_safety_state_report.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# WOMEN'S SAFETY AWARENESS
# --------------------------------------------------

elif page == "Women's Safety Awareness":

    st.title("💜 Women's Safety Awareness")

    st.write("""
    This section provides general safety-awareness information.
    It is separate from the Machine Learning predictions and
    should not be interpreted as a live safety assessment.
    """)

    st.subheader("Before travelling")

    checklist = [
        "Share your travel plan with someone you trust.",
        "Keep your phone charged.",
        "Save important emergency contacts.",
        "Check your transport arrangements.",
        "Prefer well-lit and populated routes when possible.",
        "Keep access to a trusted contact during your journey."
    ]

    for item in checklist:
        st.checkbox(item)

    st.divider()

    st.subheader("Emergency Contacts in India")

    emergency_contacts = pd.DataFrame({
        "Service": [
            "Emergency Response Support",
            "Women Helpline",
            "Police"
        ],
        "Contact Number": [
            "112",
            "181",
            "112"
        ]
    })

    st.dataframe(
        emergency_contacts,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Emergency number availability and service coverage can vary. "
        "For immediate danger in India, call 112."
    )

    st.divider()

    st.subheader("Remember")

    st.success("""
    Your safety matters. Data can help us understand broad patterns,
    but no crime-rate graph or Machine Learning model can guarantee
    that a particular place is safe or unsafe.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Women Safety Analytics | Machine Learning Project | India"
)

st.caption(
    "Analysis is based on historical reported crime data at "
    "state/UT level. Results are exploratory and are not real-time "
    "risk assessments or individual safety predictions."
)
