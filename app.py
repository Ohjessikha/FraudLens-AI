
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from predict import predict_transaction
from genai_explainer import generate_explanation

DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# 2. Custom aesthetic styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --purple: #7047EB;
        --purple-dark: #4C2CB5;
        --lavender: #F5F2FF;
        --teal: #159D91;
        --pink: #E65F91;
        --ink: #292447;
        --muted: #77738F;
        --white: #FFFFFF;
    }

    .stApp {
        background: linear-gradient(145deg, #F8F6FF 0%, #F2F7FF 55%, #F4FBF9 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.7px;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .hero {
        background: linear-gradient(120deg, #5C3BCB 0%, #7956EE 58%, #9B7AF8 100%);
        padding: 32px 34px;
        border-radius: 25px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(91, 59, 203, 0.18);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -60px;
        top: -105px;
        border: 34px solid rgba(255,255,255,0.10);
        border-radius: 50%;
    }

    .hero-brand {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #E7DFFF;
        margin-bottom: 14px;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(30px, 4vw, 43px);
        line-height: 1.15;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #EEE9FF;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
        max-width: 670px;
    }

    .hero-pill {
        display: inline-block;
        margin-top: 20px;
        padding: 7px 12px;
        border-radius: 30px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.25);
        color: white;
        font-size: 12px;
        font-weight: 600;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        color: var(--ink);
        margin: 12px 0 4px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 17px;
    }

    .metric-label {
        font-size: 12px;
        color: #716C8B;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(23px, 2.5vw, 31px);
        font-weight: 800;
        color: var(--ink);
        overflow-wrap: anywhere;
    }

    .metric-icon {
        font-size: 23px;
        margin-bottom: 13px;
    }

    .metric-card {
        background: white;
        border: 1px solid #ECE8FA;
        border-radius: 20px;
        padding: 22px;
        min-height: 155px;
        box-shadow: 0 5px 18px rgba(60, 44, 115, 0.045);
    }

    .metric-card.purple {
        border-top: 4px solid #7956EE;
    }

    .metric-card.pink {
        border-top: 4px solid #E65F91;
    }

    .metric-card.teal {
        border-top: 4px solid #159D91;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid #EAE5F7;
        border-radius: 20px;
        box-shadow: 0 6px 22px rgba(60, 44, 115, 0.045);
    }

    .stButton > button {
        background: linear-gradient(100deg, #7047EB, #8B68F2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 22px;
        min-height: 48px;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(112, 71, 235, 0.20);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        color: white;
        border: none;
        background: linear-gradient(100deg, #5934CF, #7956EE);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(112, 71, 235, 0.26);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 11px;
        border-color: #DDD5F6;
        background: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #EAE5F7;
        border-radius: 13px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #EAE5F7;
        border-radius: 13px;
        background: white;
    }

    .result-chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 25px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .chip-green {
        background: #E1F7EF;
        color: #087D60;
    }

    .chip-pink {
        background: #FCE6EF;
        color: #B73567;
    }

    .chip-purple {
        background: #EEE8FF;
        color: #633BC6;
    }

    .ai-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #4C2CB5;
    }

    .footer {
        text-align: center;
        color: #89849F;
        font-size: 12px;
        padding: 22px 0 0 0;
        line-height: 1.8;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 24px 21px;
            border-radius: 19px;
        }

        .metric-card {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset must contain a 'Class' target column.")
    st.stop()

# --------------------------------------------------
# 4. Hero section
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-brand">🛡️ FRAUDLENS AI</div>
        <h1>See beyond the numbers.</h1>
        <p>
            Intelligent transaction fraud detection powered by Machine
            Learning, with clear, AI-generated explanations for every
            prediction.
        </p>
        <div class="hero-pill">✦ MACHINE LEARNING &nbsp; + &nbsp; GENAI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Educational prototype using a public credit-card benchmark dataset. "
    "This is not a production banking system, and a prediction is not proof of fraud."
)

# --------------------------------------------------
# 5. Dataset statistics
# --------------------------------------------------
total_transactions = len(df)
fraud_count = int(df["Class"].sum())
genuine_count = total_transactions - fraud_count

st.markdown(
    '<div class="section-heading">Your dataset at a glance</div>'
    '<div class="section-subtitle">'
    'A quick overview of the transactions available for analysis.'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        f"""
        <div class="metric-card purple">
            <div class="metric-icon">💳</div>
            <div class="metric-label">TOTAL TRANSACTIONS</div>
            <div class="metric-value">{total_transactions:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card pink">
            <div class="metric-icon">🚨</div>
            <div class="metric-label">LABELLED FRAUD CASES</div>
            <div class="metric-value">{fraud_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card teal">
            <div class="metric-icon">✓</div>
            <div class="metric-label">NOT LABELLED FRAUD</div>
            <div class="metric-value">{genuine_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------
# 6. Transaction analysis
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">🔎 Analyze a transaction</div>'
    '<div class="section-subtitle">'
    'Choose a transaction and let the ML model estimate its fraud probability.'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("#### Select your transaction")

        row_number = st.number_input(
            "Dataset row number",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
            help="Rows start at 0. The selected row's actual label is not used as a model input.",
        )

        selected_row = df.iloc[int(row_number)]
        actual_label = int(selected_row["Class"])

        transaction = selected_row.drop(labels=["Class"]).to_dict()

        st.caption(
            f"Transaction row {int(row_number):,} selected "
            f"• {len(transaction)} model input features"
        )

        with st.expander("View transaction features"):
            st.dataframe(
                pd.DataFrame([transaction]),
                use_container_width=True,
                hide_index=True,
            )

        predict_clicked = st.button(
            "✦  Predict transaction",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.markdown("#### What happens next?")
        st.markdown(
            """
            <div style="
                background:linear-gradient(135deg,#F3EEFF,#F0FAF8);
                padding:20px;
                border-radius:16px;
                margin:8px 0 15px 0;
            ">
                <div style="font-size:27px;">🧠</div>
                <div style="font-size:16px;font-weight:800;color:#35265F;">
                    Machine Learning analysis
                </div>
                <div style="font-size:13px;color:#77738F;line-height:1.7;">
                    The trained model estimates the probability of fraud from
                    the transaction features.
                </div>
                <hr style="border:0;border-top:1px solid #DDD5F6;margin:15px 0;">
                <div style="font-size:27px;">✨</div>
                <div style="font-size:16px;font-weight:800;color:#35265F;">
                    GenAI explanation
                </div>
                <div style="font-size:13px;color:#77738F;line-height:1.7;">
                    Local Ollama generates a readable explanation of the
                    model output. It does not independently verify fraud.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# 7. Prediction and GenAI results
# --------------------------------------------------
if predict_clicked:
    try:
        result = predict_transaction(transaction)

        probability = float(result["fraud_probability"])
        prediction = str(result["prediction"])
        risk = str(result["risk_category"])

        st.write("")
        st.markdown(
            '<div class="section-heading">📊 Analysis results</div>'
            '<div class="section-subtitle">'
            'Results produced by your trained model for the selected transaction.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            is_flagged = "fraud" in prediction.lower() and (
                "not" not in prediction.lower()
            )

            chip_class = "chip-pink" if is_flagged else "chip-green"
            chip_text = "⚠️ FLAGGED FOR REVIEW" if is_flagged else "✓ NOT FLAGGED"

            st.markdown(
                f'<span class="result-chip {chip_class}">{chip_text}</span>',
                unsafe_allow_html=True,
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Estimated fraud probability",
                    f"{probability:.2%}",
                )

            with p2:
                st.metric("Model prediction", prediction)

            with p3:
                st.metric("Illustrative risk category", risk)

            st.progress(
                max(0.0, min(1.0, probability)),
                text=f"Estimated fraud probability: {probability:.2%}",
            )

            st.caption(
                f"Classification threshold: {result['threshold']:.0%}. "
                "Risk categories are illustrative and are not validated banking thresholds."
            )

        st.write("")

        with st.container(border=True):
            st.markdown(
                """
                <div class="ai-heading">✨ GenAI Insights</div>
                <div style="font-size:13px;color:#77738F;margin:5px 0 16px 0;">
                    Your prediction, explained in simple language by local AI.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.spinner("Generating your explanation with Ollama..."):
                explanation = generate_explanation(result)

            st.write(explanation)

        with st.container(border=True):
            st.markdown("#### 🧾 Dataset label comparison")

            if actual_label == 1:
                st.markdown(
                    '<span class="result-chip chip-pink">'
                    'DATASET LABEL: FRAUD</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="result-chip chip-green">'
                    'DATASET LABEL: NOT FRAUD</span>',
                    unsafe_allow_html=True,
                )

            predicted_fraud = int(
                str(result["predicted_class"]) == "1"
                or str(result["predicted_class"]).lower() == "fraud"
            )

            if predicted_fraud == actual_label:
                st.success(
                    "The model prediction matches this dataset label."
                )
            else:
                st.warning(
                    "The model prediction differs from this dataset label."
                )

            st.caption(
                "The actual label is displayed only after prediction for "
                "educational comparison. It is not passed to the model."
            )

    except Exception as error:
        st.error(f"Could not predict this transaction: {error}")

# --------------------------------------------------
# 8. Dataset overview
# --------------------------------------------------
st.write("")
st.markdown(
    '<div class="section-heading">📈 Dataset overview</div>'
    '<div class="section-subtitle">'
    'Explore the class imbalance and inspect a sample of the dataset.'
    '</div>',
    unsafe_allow_html=True,
)

chart_col, info_col = st.columns([1.2, 1], gap="large")

with chart_col:
    with st.container(border=True):
        st.markdown("#### Transaction class distribution")

        class_counts = (
            df["Class"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        chart_data = pd.DataFrame(
            {
                "Transaction type": ["Not fraud", "Fraud"],
                "Number of transactions": [
                    int(class_counts.loc[0]),
                    int(class_counts.loc[1]),
                ],
            }
        ).set_index("Transaction type")

        st.bar_chart(
            chart_data,
            color=["#7956EE"],
            use_container_width=True,
        )

        fraud_percentage = (
            fraud_count / total_transactions * 100
            if total_transactions
            else 0
        )

        st.caption(
            f"Fraud-labelled transactions make up {fraud_percentage:.3f}% "
            "of the dataset. The strong imbalance is an important modelling challenge."
        )

with info_col:
    with st.container(border=True):
        st.markdown("#### What the data tells us")

        st.markdown(
            f"""
            <div style="padding:8px 0;">
                <div style="color:#77738F;font-size:12px;">NOT LABELLED FRAUD</div>
                <div style="font-size:25px;font-weight:800;color:#159D91;">
                    {genuine_count:,}
                </div>
                <div style="background:#E2F5F1;height:7px;border-radius:8px;">
                    <div style="background:#159D91;height:7px;width:99.8%;border-radius:8px;"></div>
                </div>
            </div>
            <div style="padding:14px 0 5px 0;">
                <div style="color:#77738F;font-size:12px;">LABELLED FRAUD</div>
                <div style="font-size:25px;font-weight:800;color:#E65F91;">
                    {fraud_count:,}
                </div>
                <div style="background:#FCE6EF;height:7px;border-radius:8px;">
                    <div style="background:#E65F91;height:7px;width:{max(fraud_percentage, 0.5):.3f}%;border-radius:8px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "A model can miss fraud or flag non-fraud transactions. "
            "Performance must be evaluated using multiple metrics."
        )

with st.container(border=True):
    st.markdown("#### Recent dataset rows")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        🛡️ <b>FraudLens AI</b> · Machine Learning + Generative AI
        <br>
        Built for educational demonstration. Predictions are estimates,
        not proof of fraud or financial advice.
        <br>
        Dataset uses credit-card benchmark data, not debit-card-specific data.
    </div>
    """,
    unsafe_allow_html=True,
)
