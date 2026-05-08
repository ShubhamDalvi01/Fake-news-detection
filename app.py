# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== MODEL LOADING =====================

@st.cache_resource
def load_tfidf_components():
    log_reg = joblib.load("tfidf_logreg_balanced (1).pkl")
    vectorizer = joblib.load("tfidf_vectorizer_balanced (1).pkl")
    return log_reg, vectorizer

log_reg_balanced, tfidf_vectorizer = load_tfidf_components()

# ===================== PREDICTION FUNCTION =====================

def predict_tfidf(text: str, threshold: float):
    if not isinstance(text, str):
        text = str(text)

    X_vec = tfidf_vectorizer.transform([text])
    proba = log_reg_balanced.predict_proba(X_vec)[0]  # [P(fake), P(real)]
    real_prob = float(proba[1])
    label = "REAL" if real_prob >= threshold else "FAKE"
    return label, real_prob

# ===================== SAMPLE STATEMENTS =====================

SAMPLE_STATEMENTS = [
    # Likely REAL
    "Government announces a new scheme for farmers in Maharashtra with immediate subsidy benefits.",
    "Scientists discover a new exoplanet with conditions similar to Earth, raising hopes for future colonization.",
    "The Reserve Bank of India increases repo rate by 25 basis points to control inflation.",
    "Local authorities launch a road safety campaign to reduce accidents during monsoon season.",
    "Health ministry approves a new vaccine after successful phase-3 trials across multiple states.",
    "Startup launches an AI-powered platform to help small businesses automate customer support.",
    "State government declares a public holiday due to severe heatwave conditions.",
    "University researchers publish a paper on using deep learning to detect cyber attacks in real time.",
    "New metro line in the city reduces average commute time by 30 minutes for daily passengers.",
    "Government launches a portal to help citizens verify the authenticity of news articles.",

    # Clearly FAKE / exaggerated
    "Celebrity claims to have cured cancer overnight using a secret herbal tea recipe.",
    "Study shows that drinking only water for seven days permanently boosts IQ by 50 points.",
    "Social media post claims that all ATMs will stop working after midnight due to a global reset.",
    "Article alleges that mobile networks are secretly reading users' minds through 5G towers.",
    "Report suggests that eating chocolate every hour guarantees weight loss without exercise.",
    "Fake news article claims that all engineering exams have been cancelled permanently.",
    "Blog post says that staring at the sun for 10 minutes a day can improve eyesight naturally.",
    "Online rumor spreads that a popular messaging app will start charging 5000 rupees per month.",
    "Clickbait headline promises that one simple trick can double your bank balance overnight.",
    "Post claims that a man teleported from Mumbai to Delhi in 3 seconds using a new smartphone app.",
    "A new smartphone update is rumored to make phones explode if used more than six hours a day.",
    "Viral post claims that drinking petrol can boost energy and cure all diseases.",
    "Article says a man became invisible after eating a special type of mushroom.",
    "Rumor spreads that sleeping only 10 minutes a day guarantees success in all exams.",
    "Fake report claims the moon will fall closer to Earth next week, causing all oceans to vanish.",
    "Headline states that a student coded an AI that can predict the future with 100% accuracy.",
    "Post claims that a secret button in every car can make it fly during traffic jams.",
    "Blog alleges that watching TV in 8K resolution can increase human lifespan by 30 years.",
    "News claims that a city removed all traffic signals, yet accidents dropped to zero overnight.",
    "Rumor says a new smartwatch can read thoughts and send them as messages without typing.",
]

if "history" not in st.session_state:
    st.session_state.history = []

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

with st.sidebar:
    st.title("📰 Fake News Detector")
    st.markdown(
        "Backend: **TF-IDF + Logistic Regression (balanced)**\n\n"
        "- 1 = Real news\n"
        "- 0 = Fake news\n\n"
        "Adjust the decision threshold below: higher threshold → stricter REAL."
    )

    threshold = st.slider(
        "Decision threshold for REAL (P(REAL) ≥ threshold → REAL)",
        min_value=0.3,
        max_value=0.9,
        value=0.7,
        step=0.05,
    )

    st.caption(
        "Example: threshold 0.7 means only predictions with P(REAL) ≥ 0.70 "
        "are labeled REAL; others are labeled FAKE."
    )

st.title("📰 Fake News Detection Dashboard")

tab_classify, tab_batch, tab_insights = st.tabs(
    ["🔎 Classify News", "📚 Batch Samples", "📊 Model Insights"]
)

# ===================== TAB 1: CLASSIFY NEWS =====================

with tab_classify:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Single / Sample News Statements")

        choice = st.selectbox(
            "Choose a sample statement or use your own:",
            ["Custom input"] + [f"{i+1}. {txt}" for i, txt in enumerate(SAMPLE_STATEMENTS)],
            index=0,
        )

        if choice == "Custom input":
            default_text = SAMPLE_STATEMENTS[0]
        else:
            idx = int(choice.split(".")[0]) - 1
            default_text = SAMPLE_STATEMENTS[idx]

        input_text = st.text_area(
            "News text (you can edit this):",
            value=default_text,
            height=200,
        )

        if st.button("Classify", type="primary"):
            if not input_text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing text..."):
                    label, proba = predict_tfidf(input_text, threshold=threshold)

                st.session_state.history.append(
                    {"text": input_text, "label": label, "proba": proba, "threshold": threshold}
                )

                st.markdown("### Prediction")
                if label == "REAL":
                    st.success(f"Predicted label: **{label}**")
                else:
                    st.error(f"Predicted label: **{label}**")

                st.write(f"Model confidence (Real probability): {proba:.3f}")
                st.write(f"Current decision threshold: {threshold:.2f}")

                fig_prob, ax_prob = plt.subplots(figsize=(4, 0.4))
                ax_prob.barh(
                    [""],
                    [proba * 100],
                    color="#4CAF50" if label == "REAL" else "#F44336"
                )
                ax_prob.set_xlim(0, 100)
                ax_prob.set_xlabel("Real probability (%)")
                ax_prob.set_yticks([])
                st.pyplot(fig_prob)

                st.caption(
                    "Prediction using balanced TF-IDF + Logistic Regression model "
                    f"with threshold {threshold:.2f} for REAL."
                )

    with col_right:
        st.subheader("Session Summary")

        total_preds = len(st.session_state.history)
        if total_preds == 0:
            st.caption("No predictions yet. Run a classification to see stats.")
        else:
            real_count = sum(1 for h in st.session_state.history if h["label"] == "REAL")
            fake_count = total_preds - real_count

            st.metric("Total predictions", total_preds)
            st.metric("REAL predictions", real_count)
            st.metric("FAKE predictions", fake_count)

            labels_pie = ["REAL", "FAKE"]
            sizes_pie = [real_count, fake_count]
            fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
            colors = ["#4CAF50", "#F44336"]
            ax_pie.pie(
                sizes_pie,
                labels=labels_pie,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors
            )
            ax_pie.axis("equal")
            ax_pie.set_title("Session Predictions")
            st.pyplot(fig_pie)

        if total_preds > 0:
            st.markdown("#### Recent Predictions")
            last_items = st.session_state.history[-5:][::-1]
            st.table(
                {
                    "Text (truncated)": [
                        h["text"][:80] + "..." if len(h["text"]) > 80 else h["text"]
                        for h in last_items
                    ],
                    "Label": [h["label"] for h in last_items],
                    "Real prob": [f"{h['proba']:.3f}" for h in last_items],
                    "Threshold": [f"{h['threshold']:.2f}" for h in last_items],
                }
            )

# ===================== TAB 2: BATCH SAMPLES (COMPACT) =====================

with tab_batch:
    st.subheader("Batch Classification on Sample Statements")
    st.markdown(
        "Run the model on multiple sample statements to see overall behavior "
        "for the **current threshold**."
    )

    n_samples = st.slider(
        "Number of sample statements",
        min_value=5,
        max_value=len(SAMPLE_STATEMENTS),
        value=15,
        step=5
    )

    if st.button("Run Batch", type="primary", key="batch_button"):
        texts = SAMPLE_STATEMENTS[:n_samples]
        preds = []
        probs = []

        for tx in texts:
            label, proba = predict_tfidf(tx, threshold=threshold)
            preds.append(label)
            probs.append(proba)

        df_batch = pd.DataFrame(
            {
                "content": texts,
                "prediction": preds,
                "real_prob": [f"{p:.3f}" for p in probs],
            }
        )

        real_count = sum(1 for p in preds if p == "REAL")
        fake_count = len(preds) - real_count

        col_summary, col_chart = st.columns([1, 1])

        with col_summary:
            st.markdown("### Batch Summary")
            st.write(f"Total samples: **{len(texts)}**")
            st.write(f"REAL predictions: **{real_count}**")
            st.write(f"FAKE predictions: **{fake_count}**")
            st.write(f"Threshold used: **{threshold:.2f}**")

        with col_chart:
            fig_batch_pie, ax_batch_pie = plt.subplots(figsize=(2, 2))
            ax_batch_pie.pie(
                [real_count, fake_count],
                labels=["REAL", "FAKE"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4CAF50", "#F44336"]
            )
            ax_batch_pie.axis("equal")
            ax_batch_pie.set_title("Batch Predictions")
            st.pyplot(fig_batch_pie)

        st.markdown("### Batch Details (first 10 rows)")
        df_show = df_batch.copy().head(10)
        df_show["content"] = df_show["content"].str.slice(0, 80) + "..."
        st.dataframe(df_show, height=220)

# ===================== TAB 3: MODEL INSIGHTS (TF-IDF vs LSTM) =====================

with tab_insights:
    st.subheader("Model Performance Comparison (TF-IDF + LR vs LSTM)")

    st.markdown(
        "Metrics below are taken from your Colab experiments for:\n\n"
        "- **TF-IDF + Logistic Regression (balanced)** (deployed model)\n"
        "- **LSTM with embeddings** (comparison model)\n"
    )

    # TODO: Replace these with your real numbers from Colab
    ACC_TFIDF = 0.90
    TFIDF_MACRO_PREC = 0.90
    TFIDF_MACRO_REC = 0.90
    TFIDF_MACRO_F1 = 0.90

    ACC_LSTM = 0.75
    LSTM_MACRO_PREC = 0.76
    LSTM_MACRO_REC = 0.73
    LSTM_MACRO_F1 = 0.73

    # Table
    metrics_data = {
        "Model": ["TF-IDF + LogReg", "Embedding + LSTM"],
        "Accuracy": [f"{ACC_TFIDF:.4f}", f"{ACC_LSTM:.4f}"],
        "Macro Precision": [f"{TFIDF_MACRO_PREC:.4f}", f"{LSTM_MACRO_PREC:.4f}"],
        "Macro Recall": [f"{TFIDF_MACRO_REC:.4f}", f"{LSTM_MACRO_REC:.4f}"],
        "Macro F1-score": [f"{TFIDF_MACRO_F1:.4f}", f"{LSTM_MACRO_F1:.4f}"],
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Accuracy Comparison")

        models = ["TF-IDF + LogReg", "Embedding + LSTM"]
        accuracies = [ACC_TFIDF, ACC_LSTM]

        fig_bar, ax_bar = plt.subplots(figsize=(3.2, 3))
        bars = ax_bar.bar(models, accuracies, color=["#2196F3", "#FF9800"])
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("Accuracy")
        ax_bar.set_title("Test Accuracy (Colab)")
        ax_bar.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, accuracies):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        st.pyplot(fig_bar)

    with col_b:
        st.markdown("#### Macro F1-score Comparison")

        f1_scores = [TFIDF_MACRO_F1, LSTM_MACRO_F1]

        fig_f1, ax_f1 = plt.subplots(figsize=(3.2, 3))
        bars_f1 = ax_f1.bar(models, f1_scores, color=["#4CAF50", "#9C27B0"])
        ax_f1.set_ylim(0, 1.0)
        ax_f1.set_ylabel("Macro F1-score")
        ax_f1.set_title("Macro F1-score (Colab)")
        ax_f1.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars_f1, f1_scores):
            ax_f1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        st.pyplot(fig_f1)

    st.markdown(
        "- TF-IDF + LR is used in the live app because it is lighter and easier to deploy.\n"
        "- LSTM gives a deep learning baseline for comparison but is more expensive to train.\n"
        "- You can discuss these trade-offs in your viva as model selection reasoning."
    )
