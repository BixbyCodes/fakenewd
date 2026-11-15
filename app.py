import streamlit as st
import pandas as pd

from src.predict import predict_news

st.set_page_config(
    page_title="EvidenceAware | Fake News Detection",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background: #0b1020; color: #e5e7eb;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {
    padding: 28px; border: 1px solid #29334d; border-radius: 18px;
    background: linear-gradient(120deg,#151f38,#111827);
    margin-bottom: 24px;
}
.hero h1 {font-size: 2.3rem; margin-bottom: 5px;}
.hero p {color: #aab6cc; margin: 0;}
.metric {
    background: #151d30; border: 1px solid #29334d;
    padding: 18px; border-radius: 14px; min-height: 110px;
}
.metric-label {color:#9aa9c2; font-size: .85rem;}
.metric-value {font-size: 1.8rem; font-weight: 700; margin-top: 7px;}
.section {
    color:#e5e7eb; font-size:1.2rem; font-weight:650;
    margin: 24px 0 12px;
}
div.stButton > button {
    background: #6477f5; color: white; border: 0;
    border-radius: 9px; padding: .65rem 1.5rem;
    font-weight: 650;
}
div.stButton > button:hover {background:#5265df; color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔎 EvidenceAware</h1>
    <p>AI-assisted fake news detection and evidence analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Analyze a news article")
article = st.text_area(
    "Paste the headline or article text",
    height=190,
    placeholder="Paste the complete news article here..."
)

analyze = st.button("Analyze article", type="primary")

if analyze:
    if not article.strip():
        st.warning("Please enter some news text first.")
        st.stop()

    with st.spinner("Running the classifier..."):
        try:
            result = predict_news(article)
            st.session_state["prediction"] = result
            st.session_state["article"] = article
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.success("Classification completed.")

if "prediction" in st.session_state:
    result = st.session_state["prediction"]
    article = st.session_state["article"]

    st.markdown('<div class="section">01 · Classification results</div>',
                unsafe_allow_html=True)

    prediction = result["prediction"]
    color = "#ff646e" if prediction == "FAKE" else "#36c98f"

    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">MODEL PREDICTION</div>
        <div class="metric-value" style="color:{color};font-size:2.2rem">
            {prediction}
        </div>
        <div class="metric-label">Based on the trained ISOT text classifier</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    metrics = [
        ("Model confidence", f'{result["confidence"]}%'),
        ("Fake probability", f'{result["fake_probability"]}%'),
        ("Real probability", f'{result["real_probability"]}%')
    ]

    for col, (label, value) in zip([c1, c2, c3], metrics):
        with col:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.progress(result["fake_probability"] / 100,
                text="Estimated probability of FAKE")

    st.markdown('<div class="section">02 · Submitted article</div>',
                unsafe_allow_html=True)
    st.write(article)

    st.markdown('<div class="section">03 · Event and entity extraction</div>',
                unsafe_allow_html=True)

    event_data = None
    try:
        from src.event_parser import parse_event
        with st.spinner("Extracting event details with the language model..."):
            event_data = parse_event(article)

        st.json(event_data)

        entities = event_data.get("entities", [])
        if entities:
            st.markdown("**Extracted entities**")
            entity_rows = []
            for item in entities:
                if isinstance(item, dict):
                    entity_rows.append({
                        "Entity": item.get("name", ""),
                        "Type": item.get("type", ""),
                        "Role": item.get("role", "")
                    })
            if entity_rows:
                st.dataframe(pd.DataFrame(entity_rows),
                             use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Event extraction unavailable: {e}")

    st.markdown('<div class="section">04 · Evidence verification</div>',
                unsafe_allow_html=True)

    try:
        from src.evidence_verifier import analyze_evidence

        normalized = (
            event_data.get("normalized_claim")
            if event_data else None
        )

        with st.spinner("Retrieving and verifying evidence sentences..."):
            evidence_report = analyze_evidence(
                article,
                normalized_claim=normalized,
                page_limit=5
            )

        results = evidence_report["results"]

        counts = {
            label: sum(1 for item in results if item["label"] == label)
            for label in ["SUPPORTS", "REFUTES", "INSUFFICIENT"]
        }

        st.caption(
            f'{evidence_report["evidence_count"]} sentences retrieved · '
            f'{evidence_report["verified_count"]} sentences classified'
        )

        a, b, c = st.columns(3)
        for col, label, color in [
            (a, "SUPPORTS", "#36c98f"),
            (b, "REFUTES", "#ff646e"),
            (c, "INSUFFICIENT", "#aab6cc")
        ]:
            with col:
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">
                        {counts[label]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if results:
            st.markdown("#### Retrieved evidence")

            for index, item in enumerate(results, 1):
                label = item["label"]
                icon = {
                    "SUPPORTS": "🟢",
                    "REFUTES": "🔴",
                    "INSUFFICIENT": "⚪"
                }[label]

                with st.expander(
                    f'{icon} {label} · {item["page_id"]} · '
                    f'Sentence {item["sentence_id"]}'
                ):
                    st.write("**Evidence sentence**")
                    st.write(item["sentence"])

                    st.write("**LLM explanation**")
                    st.write(item["explanation"])

                    st.caption(
                        f'FEVER Wikipedia snapshot · '
                        f'{item["page_id"]} · '
                        f'Sentence ID: {item["sentence_id"]}'
                    )

                    wiki_title = item["page_id"].replace("_", " ")
                    st.markdown(
                        f'[Open Wikipedia page](https://en.wikipedia.org/wiki/'
                        f'{item["page_id"]})'
                    )
        else:
            st.warning(
                "No sentence-level evidence was verified. "
                "This does not mean the article is false."
            )

        st.caption(
            "Evidence labels are LLM assessments of retrieved sentences. "
            "They are not independently verified ground truth."
        )

    except Exception as e:
        st.error(f"Evidence verification failed: {e}")


    if "counts" in locals():
        st.markdown('<div class="section">05 · Final evidence assessment</div>',
                    unsafe_allow_html=True)

        supports = counts["SUPPORTS"]
        refutes = counts["REFUTES"]
        insufficient = counts["INSUFFICIENT"]

        if refutes > 0 and supports > 0:
            status = "CONFLICTING EVIDENCE"
            color = "#f5b942"
            explanation = "Retrieved sentences contain both supporting and refuting assessments."
        elif refutes > supports:
            status = "CONTRADICTED BY RETRIEVED EVIDENCE"
            color = "#ff646e"
            explanation = "More retrieved sentences were assessed as refuting than supporting the claim."
        elif supports > refutes:
            status = "SUPPORTED BY RETRIEVED EVIDENCE"
            color = "#36c98f"
            explanation = "More retrieved sentences were assessed as supporting than refuting the claim."
        else:
            status = "INSUFFICIENT EVIDENCE"
            color = "#aab6cc"
            explanation = "The retrieved evidence does not establish a clear direction."

        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">EVIDENCE-BASED ASSESSMENT</div>
            <div class="metric-value" style="color:{color};font-size:1.6rem">
                {status}
            </div>
            <div class="metric-label">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.metric("Classifier prediction", result["prediction"])
        with right:
            st.metric("Classifier confidence", f'{result["confidence"]}%')

        st.warning(
            "The evidence assessment is based on retrieved sentences and "
            "LLM-generated labels. It is not a calibrated probability of truth. "
            "The classifier and evidence assessment are separate signals."
        )

    st.markdown('<div class="section">06 · Interpretation and limitations</div>',
                unsafe_allow_html=True)

    st.info(
        "The FAKE/REAL result comes from a text classifier trained on the "
        "ISOT dataset. Event extraction and Wikipedia retrieval provide "
        "additional context; they do not yet independently verify the claim. "
        "Treat uncertain or high-impact results as requiring human review."
    )

    with st.expander("Technical details"):
        st.json({
            "classifier": "ISOT Logistic Regression",
            "vectorizer": "Saved ISOT TF-IDF model",
            "label_mapping": {"0": "REAL", "1": "FAKE"},
            "event_extraction": "Groq LLM, when available",
            "evidence_index": "FEVER Wikipedia title index",
            "prediction": result
        })

st.markdown("---")
st.caption("EvidenceAware · Research prototype · Predictions are not fact-checking verdicts.")
