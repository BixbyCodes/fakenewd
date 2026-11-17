"""EvidenceAware — Streamlit front end.

Run locally:  streamlit run app.py
"""

import hashlib
import html
import logging
import os

import streamlit as st

# Streamlit Community Cloud stores secrets in st.secrets. Bridge them into the
# environment before any project module reads configuration. Values are never
# printed or rendered.
for _name in ("GROQ_API_KEY", "GROQ_MODEL", "EAFND_WIKI_INDEX", "EAFND_EVIDENCE_BACKEND",
              "WIKIPEDIA_USER_AGENT"):
    if not os.getenv(_name):
        try:
            if _name in st.secrets:
                os.environ[_name] = str(st.secrets[_name])
        except Exception:  # no secrets file configured
            pass

from src.analysis import InputError, capability_status, run_analysis  # noqa: E402
from src.config import MAX_INPUT_CHARS  # noqa: E402
from src.predict import load_models  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("eafnd.app")

st.set_page_config(
    page_title="EvidenceAware · Evidence-aware fake news analysis",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

EXAMPLES = {
    "Claim likely contradicted by Wikipedia": {
        "headline": "",
        "body": "Telemundo is an English-language television network.",
    },
    "Claim likely supported by Wikipedia": {
        "headline": "",
        "body": "Fox 2000 Pictures released the film Soul Food.",
    },
    "Claim with little encyclopedic evidence": {
        "headline": "",
        "body": "The Springfield city council approved a new budget for bicycle lanes "
                "at its meeting last Tuesday evening.",
    },
    "News-style article": {
        "headline": "Apollo 11 astronauts return after first crewed Moon landing",
        "body": "Astronauts Neil Armstrong, Buzz Aldrin and Michael Collins returned to Earth "
                "on Thursday after a mission that placed the first humans on the surface of "
                "the Moon. Armstrong and Aldrin spent about two and a half hours walking on the "
                "lunar surface, collecting rock samples and setting up experiments, while "
                "Collins remained in orbit aboard the command module Columbia. The crew "
                "splashed down in the Pacific Ocean, where they were recovered by the aircraft "
                "carrier USS Hornet. NASA officials said the astronauts would spend several "
                "weeks in quarantine as a precaution before being reunited with their families.",
    },
    "Ambiguous viral claim": {
        "headline": "",
        "body": "Posts shared widely online say that doctors have discovered a common kitchen "
                "spice can completely cure cancer within weeks, but that the treatment is "
                "being kept quiet by pharmaceutical companies.",
    },
}

LABEL_STYLE = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "INSUFFICIENT": "insufficient",
    "FAKE": "refutes",
    "REAL": "supports",
    "UNCERTAIN": "insufficient",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&display=swap');
:root {
  --bg:#F6F4EF; --surface:#FFFFFF; --ink:#1F2328; --muted:#6B665E; --line:#E3DED4;
  --green:#3B7353; --green-bg:#EAF2EC; --red:#A2413A; --red-bg:#F7E8E5;
  --amber:#946112; --amber-bg:#FAF0DB; --slate:#3D4A55; --slate-bg:#EEF0F1;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3, .ea-serif { font-family: 'Source Serif 4', Georgia, serif !important; color: var(--ink); letter-spacing: -0.01em; }
.ea-mast { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
  border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:6px; }
.ea-brand { font-family:'Source Serif 4', Georgia, serif; font-size:1.9rem; font-weight:600; line-height:1.1; }
.ea-brand span { color: var(--muted); font-weight:500; }
.ea-tag { color: var(--muted); font-size:.95rem; margin-top:4px; }
.ea-proto { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:3px 10px; white-space:nowrap; }
.ea-status { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 4px; }
.ea-chip { font-size:.78rem; border-radius:999px; padding:3px 10px; border:1px solid var(--line); background:var(--surface); color:var(--ink); }
.ea-chip.on::before { content:"●"; color:var(--green); margin-right:6px; font-size:.7rem; }
.ea-chip.off::before { content:"○"; color:var(--muted); margin-right:6px; font-size:.7rem; }
.ea-section { margin:34px 0 10px; display:flex; align-items:baseline; gap:10px; }
.ea-section .num { font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums; }
.ea-section .ttl { font-family:'Source Serif 4', Georgia, serif; font-size:1.35rem; font-weight:600; }
.ea-card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:12px; }
.ea-kicker { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:6px; }
.ea-big { font-family:'Source Serif 4', Georgia, serif; font-size:2.1rem; font-weight:600; line-height:1.1; }
.ea-sub { color:var(--muted); font-size:.88rem; margin-top:6px; }
.ea-pill { display:inline-block; font-size:.74rem; font-weight:600; letter-spacing:.04em; border-radius:4px; padding:2px 8px; }
.supports { color:var(--green); } .ea-pill.supports { background:var(--green-bg); }
.refutes { color:var(--red); } .ea-pill.refutes { background:var(--red-bg); }
.insufficient { color:var(--amber); } .ea-pill.insufficient { background:var(--amber-bg); }
.neutral { color:var(--slate); } .ea-pill.neutral { background:var(--slate-bg); }
.ea-bar { display:flex; height:12px; border-radius:6px; overflow:hidden; background:var(--line); margin:10px 0 6px; }
.ea-bar .f { background:var(--red); } .ea-bar .r { background:var(--green); }
.ea-legend { display:flex; justify-content:space-between; font-size:.85rem; color:var(--muted); font-variant-numeric:tabular-nums; }
.ea-meter { margin:8px 0 12px; }
.ea-meter .row { display:flex; justify-content:space-between; font-size:.88rem; }
.ea-meter .track { height:6px; background:var(--line); border-radius:3px; margin-top:4px; }
.ea-meter .fill { height:6px; background:var(--slate); border-radius:3px; }
.ea-note { border-left:3px solid var(--amber); background:var(--amber-bg); padding:10px 14px; border-radius:0 6px 6px 0; font-size:.88rem; margin:8px 0; color:#4A3A1A; }
.ea-info { border-left:3px solid var(--slate); background:var(--slate-bg); padding:10px 14px; border-radius:0 6px 6px 0; font-size:.88rem; margin:8px 0; color:#2C3640; }
.ea-article { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px 20px;
  max-height:300px; overflow-y:auto; white-space:pre-wrap; font-family:'Source Serif 4', Georgia, serif; font-size:1rem; line-height:1.6; }
.ea-dl { display:grid; grid-template-columns: 150px 1fr; gap:8px 16px; font-size:.93rem; }
.ea-dl dt { color:var(--muted); } .ea-dl dd { margin:0; }
.ea-counts { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:6px 0 12px; }
.ea-count { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.ea-count .n { font-family:'Source Serif 4', Georgia, serif; font-size:1.7rem; font-weight:600; }
.ea-count .l { font-size:.75rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.ea-verdict { border-left:4px solid var(--slate); }
.ea-verdict.supports { border-left-color:var(--green); } .ea-verdict.refutes { border-left-color:var(--red); }
.ea-verdict.conflict, .ea-verdict.uncertain { border-left-color:var(--amber); }
.ea-verdict h3 { font-size:1.25rem; margin:0 0 10px; color:var(--ink) !important; }
.ea-verdict ul { margin:0 0 0 1.1rem; padding:0; } .ea-verdict li { margin-bottom:6px; font-size:.93rem; color:var(--ink); }
.ea-foot { color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); margin-top:40px; padding-top:12px; }
div.stButton > button, div.stFormSubmitButton > button { border-radius:6px; font-weight:600; }
[data-testid="stExpander"] details { background:var(--surface); border-color:var(--line); border-radius:8px; }
@media (max-width: 640px) {
  .block-container { padding-left:16px; padding-right:16px; padding-top:1.2rem; }
  .ea-brand { font-size:1.55rem; } .ea-big { font-size:1.7rem; }
  .ea-dl { grid-template-columns: 1fr; gap:2px; } .ea-dl dd { margin-bottom:8px; }
  .ea-counts { gap:6px; } .ea-count { padding:10px; } .ea-count .n { font-size:1.4rem; }
}
</style>
"""


def esc(value):
    return html.escape("" if value is None else str(value))


def section(number, title):
    st.markdown(
        '<div class="ea-section"><span class="num">{:02d}</span>'
        '<span class="ttl">{}</span></div>'.format(number, esc(title)),
        unsafe_allow_html=True,
    )


def note(text, kind="note"):
    st.markdown('<div class="ea-{}">{}</div>'.format(kind, esc(text)), unsafe_allow_html=True)


def pill(label, style=None):
    style = style or LABEL_STYLE.get(str(label).upper(), "neutral")
    return '<span class="ea-pill {}">{}</span>'.format(style, esc(label))


@st.cache_resource(show_spinner="Loading trained classifiers…")
def warm_models():
    load_models()
    return True


@st.cache_data(ttl=300, show_spinner=False)
def cached_status():
    return capability_status()


def render_header(status):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ea-mast">
          <div>
            <div class="ea-brand">EvidenceAware <span>· EAFND</span></div>
            <div class="ea-tag">Classifier prediction, LLM assessment and Wikipedia evidence for a news claim, kept separate and explained.</div>
          </div>
          <div class="ea-proto">Research prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chips = [
        ("Classifier", status["classifier"], "available"),
        ("LLM analysis", status["llm"], "available"),
        ("Wikipedia evidence", status["evidence"],
         status["evidence_label"] if status["evidence"] else "unavailable"),
    ]
    st.markdown(
        '<div class="ea-status" aria-label="Component availability">{}</div>'.format("".join(
            '<span class="ea-chip {}">{}: {}</span>'.format(
                "on" if ok else "off", esc(name), esc(text if ok else "unavailable"))
            for name, ok, text in chips
        )),
        unsafe_allow_html=True,
    )
    with st.expander("About this project"):
        st.markdown(
            """
**EvidenceAware (EAFND)** is a research prototype for evidence-aware fake news analysis.
For each submission it reports four separate signals:

1. **Classifier prediction** — a TF-IDF + Linear SVM model trained on the ISOT fake news
   dataset. It learns writing patterns from ISOT; it does not check facts.
2. **LLM assessment** — `openai/gpt-oss-20b` via the Groq API (inference only; not trained
   by this project) rates plausibility, verifiability and source attribution.
3. **Event & entity extraction** — the LLM extracts the who/what/when/where of the text.
4. **Evidence assessment** — sentences are retrieved from Wikipedia (the live public
   MediaWiki API, or a local index of the 2017 FEVER snapshot when installed) and the
   LLM labels each retrieved sentence as SUPPORTS, REFUTES or INSUFFICIENT for the
   claim. The retrieved text and the LLM's label are shown separately.

The final interpretation reports where these signals agree or disagree. It is not a
fact-checking verdict and does not compute a combined truth score.
            """
        )
        if not status["llm"]:
            st.caption("LLM analysis is disabled because no Groq API key is configured on this server.")
        if not status["evidence"]:
            st.caption("Evidence retrieval is unavailable: {}".format(
                status["evidence_reason"] or "no evidence backend is configured."))


def load_example(name):
    example = EXAMPLES[name]
    st.session_state["headline_input"] = example["headline"]
    st.session_state["body_input"] = example["body"]


def render_input():
    st.markdown('<div class="ea-section"><span class="ttl">Analyze a claim or article</span></div>',
                unsafe_allow_html=True)
    with st.expander("Load an example"):
        cols = st.columns(2)
        for i, name in enumerate(EXAMPLES):
            cols[i % 2].button(name, key="example_{}".format(i), on_click=load_example,
                               args=(name,), width="stretch")

    with st.form("analyze_form", clear_on_submit=False, border=False):
        headline = st.text_input("Headline (optional)", key="headline_input",
                                 max_chars=400, placeholder="e.g. Government announces new policy")
        body = st.text_area(
            "Claim or article text",
            key="body_input",
            height=200,
            max_chars=MAX_INPUT_CHARS,
            placeholder="Paste a single claim or the full text of a news article…",
            help="Between 4 words and {:,} characters. Full articles suit the classifier best; "
                 "short factual claims suit the Wikipedia evidence check best.".format(MAX_INPUT_CHARS),
        )
        submitted = st.form_submit_button("Analyze", type="primary")
    return submitted, headline, body


def render_classifier(result):
    section(1, "Classifier prediction")
    if result is None:
        st.error("The trained classifier could not be run. The remaining sections may still be useful.")
        return
    style = LABEL_STYLE[result["prediction"]]
    acc = result.get("recorded_test_accuracy")
    st.markdown(
        """
        <div class="ea-card">
          <div class="ea-kicker">Predicted class</div>
          <div class="ea-big {style}">{pred}</div>
          <div class="ea-sub">{prob:.1f}% model probability for this class ·
            {model} trained on ISOT{acc} · {agree}</div>
        </div>
        """.format(
            style=style, pred=esc(result["prediction"]), prob=result["confidence"],
            model=esc(result["model"]),
            acc=" (held-out ISOT test accuracy {:.1f}%)".format(acc * 100) if acc else "",
            agree=esc(result["model_agreement"]),
        ),
        unsafe_allow_html=True,
    )
    for text in result["reliability_notes"]:
        note(text)

    section(2, "Probability breakdown")
    fake, real = result["fake_probability"], result["real_probability"]
    st.markdown(
        """
        <div class="ea-card">
          <div class="ea-kicker">{model} — class probabilities</div>
          <div class="ea-bar" role="img" aria-label="Fake {fake:.1f} percent, real {real:.1f} percent">
            <div class="f" style="width:{fake:.2f}%"></div><div class="r" style="width:{real:.2f}%"></div>
          </div>
          <div class="ea-legend"><span class="refutes">FAKE {fake:.1f}%</span><span class="supports">REAL {real:.1f}%</span></div>
          <div class="ea-sub">Probabilities are calibrated on ISOT validation folds. They describe how ISOT-like
          the writing is, not the probability that the claim is true.</div>
        </div>
        """.format(model=esc(result["model"]), fake=fake, real=real),
        unsafe_allow_html=True,
    )
    rows = [
        {"Model (ISOT)": item["model"], "Prediction": item["prediction"],
         "FAKE %": item["fake_probability"], "REAL %": item["real_probability"]}
        for item in result["all_models"].values()
    ]
    st.caption("All three saved ISOT classifiers on the same input (primary model listed first):")
    st.dataframe(rows, hide_index=True, width="stretch")


def render_article(report):
    section(3, "Submitted text")
    info = report["input"]
    st.markdown('<div class="ea-article">{}</div>'.format(esc(info["text"])), unsafe_allow_html=True)
    st.caption("{:,} words · {:,} characters · shown exactly as analyzed (whitespace normalized).".format(
        info["words"], info["characters"]))


def unavailable(component, what):
    message = component.get("message") or "This component did not run."
    note("{} unavailable — {}".format(what, message), "info")


def render_event(event, component):
    section(4, "Event and entity extraction")
    if event is None:
        unavailable(component, "Event extraction")
        return
    fields = [
        ("Event summary", event.get("event_summary")),
        ("Time", event.get("time")),
        ("Location", event.get("location")),
        ("Attribution", event.get("attribution")),
        ("Normalized claim", event.get("normalized_claim")),
    ]
    st.markdown(
        '<div class="ea-card"><dl class="ea-dl">{}</dl></div>'.format("".join(
            "<dt>{}</dt><dd>{}</dd>".format(esc(k), esc(v) if v else '<span class="ea-sub">not stated</span>')
            for k, v in fields
        )),
        unsafe_allow_html=True,
    )
    entities = event.get("entities") or []
    if entities:
        st.dataframe(
            [{"Entity": e["name"], "Type": e["type"], "Role": e["role"]} for e in entities],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("No named entities were extracted.")
    if event.get("uncertainties"):
        st.markdown("**Uncertainties noted by the extractor**")
        for item in event["uncertainties"]:
            st.markdown("- " + esc(item))
    st.caption(
        "Extracted by {} (LLM). The normalized claim is shown for reference only; the original "
        "text is what is classified and checked against evidence.{}".format(
            event.get("model", "the LLM"),
            " Long input was truncated for extraction." if event.get("input_truncated") else "")
    )


def meter(label, value, help_text):
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        '<div class="ea-meter"><div class="row"><span>{}</span><span>{:.2f}</span></div>'
        '<div class="track"><div class="fill" style="width:{:.0f}%"></div></div>'
        '<div class="ea-sub">{}</div></div>'.format(esc(label), float(value), pct, esc(help_text))
    )


def render_llm(llm, component):
    section(5, "LLM assessment")
    if llm is None:
        unavailable(component, "LLM assessment")
        return
    verdict = llm["verdict"].upper()
    st.markdown(
        """
        <div class="ea-card">
          <div class="ea-kicker">LLM verdict (assessment of the text only)</div>
          <div>{pill}</div>
          <div style="margin-top:14px">{m1}{m2}{m3}</div>
          <div class="ea-kicker" style="margin-top:6px">Rationale</div>
          <div style="font-size:.95rem">{rationale}</div>
        </div>
        """.format(
            pill=pill(verdict),
            m1=meter("Plausibility", llm["plausibility"], "How plausible the content seems to the LLM."),
            m2=meter("Verifiability", llm["verifiability"], "Whether the text contains checkable specifics."),
            m3=meter("Source credibility", llm["source_credibility"],
                     "Whether a source is identified in the text, not whether it is trustworthy."),
            rationale=esc(llm.get("rationale") or "No rationale returned."),
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Generated by {} via the Groq API. This is an LLM assessment without retrieved evidence; "
        "it can be wrong and does not override the classifier.".format(llm.get("model", "the LLM"))
    )


EVIDENCE_STATUS_TEXT = {
    "index_unavailable": "Evidence retrieval is not available in this environment",
    "retrieval_unavailable": "Wikipedia evidence could not be retrieved",
    "no_candidates": "No relevant Wikipedia pages were found",
    "no_sentences": "No usable evidence sentences were found",
    "retrieved_not_assessed": "Evidence retrieved but not assessed",
    "error": "Evidence retrieval failed",
}


def source_line(item, backend):
    link = '<a href="{}" target="_blank" rel="noopener">Open on Wikipedia</a>'.format(esc(item["url"]))
    if backend == "wikipedia_api":
        return ('Live Wikipedia · <strong>{}</strong> · sentence {} of the retrieved text · {}'
                .format(esc(item["title"]), item["sentence_id"], link))
    return ('FEVER Wikipedia (2017) · page <code>{}</code> · sentence ID {} · {}'
            .format(esc(item["page_id"]), item["sentence_id"], link))


def evidence_html(item, backend, show_title=False):
    title = '<strong>{}</strong> '.format(esc(item["title"])) if show_title else ""
    span = ""
    if item.get("evidence_span"):
        span = ('<div class="ea-sub" style="margin-top:6px">Deciding phrase quoted by the LLM: '
                '<em>“{}”</em></div>'.format(esc(item["evidence_span"])))
    return (
        '{title}<div class="ea-kicker">Retrieved Wikipedia sentence</div>'
        '<div class="ea-serif" style="font-size:1.02rem;margin:4px 0 6px">“{sentence}”</div>'
        '<div class="ea-sub">{source}</div>'
        '<div class="ea-kicker" style="margin-top:12px">LLM assessment of this sentence</div>'
        '<div>{pill}</div><div style="font-size:.92rem;margin-top:6px">{explanation}</div>{span}'.format(
            title=title, sentence=esc(item["sentence"]), source=source_line(item, backend),
            pill=pill(item["label"]), explanation=esc(item["explanation"] or "—"), span=span)
    )


def render_evidence(evidence, component):
    section(6, "Evidence verification")
    status = evidence.get("status")
    backend = evidence.get("backend")
    if status in ("index_unavailable", "retrieval_unavailable"):
        note(
            "Evidence retrieval did not run: {} No evidence check was performed, so nothing "
            "below should be read as verification.".format(
                evidence.get("message") or "no evidence backend is available."), "info")
        return

    st.markdown(
        '<div class="ea-sub" style="margin-bottom:8px">Claim assessed against evidence: '
        '<em>{}</em></div>'.format(esc(evidence.get("claim_assessed"))),
        unsafe_allow_html=True,
    )
    if status != "completed":
        note("{}. {}".format(EVIDENCE_STATUS_TEXT.get(status, "Evidence unavailable"),
                             evidence.get("message") or ""), "info")

    if status == "completed":
        counts = evidence["counts"]
        st.markdown(
            '<div class="ea-counts">{}</div>'.format("".join(
                '<div class="ea-count"><div class="n {}">{}</div><div class="l">{}</div></div>'.format(
                    LABEL_STYLE[label], counts[label], label.title())
                for label in ("SUPPORTS", "REFUTES", "INSUFFICIENT")
            )),
            unsafe_allow_html=True,
        )
        st.caption("Retrieval: {} sentences from {} Wikipedia pages · LLM assessment: {} labelled.".format(
            evidence["evidence_count"], len(evidence["candidate_pages"]), evidence["verified_count"]))
        if evidence.get("message"):
            st.caption(evidence["message"])
    if evidence.get("corpus"):
        st.caption("Source: {}.".format(evidence["corpus"]))
    if evidence.get("queries"):
        with st.expander("Search queries used"):
            for query in evidence["queries"]:
                st.markdown("- " + esc(query))
            st.caption("The original claim is always the first query and is the only text "
                       "evidence is assessed against.")

    if evidence.get("candidate_pages"):
        with st.expander("Candidate Wikipedia pages ({})".format(len(evidence["candidate_pages"]))):
            st.dataframe(
                [{"Page": c["title"], "Found via": c["source"], "Link": c["url"]}
                 for c in evidence["candidate_pages"]],
                hide_index=True, width="stretch",
                column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
            )

    order = {"REFUTES": 0, "SUPPORTS": 1, "INSUFFICIENT": 2}
    if status == "completed":
        items = sorted(evidence["results"], key=lambda r: order.get(r["label"], 3))
        if not items:
            st.caption("The LLM returned no valid labels for the retrieved sentences.")
        decisive = [r for r in items if r["label"] != "INSUFFICIENT"]
        if not decisive and items:
            st.caption("No retrieved sentence directly supports or contradicts the claim. "
                       "This does not mean the claim is true or false.")
        for item in decisive:
            with st.expander("{} · {} · sentence {}".format(item["label"], item["title"], item["sentence_id"]),
                             expanded=len(decisive) <= 4):
                st.markdown(evidence_html(item, backend), unsafe_allow_html=True)
        insufficient = [r for r in items if r["label"] == "INSUFFICIENT"]
        if insufficient:
            with st.expander("{} sentence(s) assessed as INSUFFICIENT".format(len(insufficient))):
                st.markdown(
                    '<hr style="margin:12px 0;border:none;border-top:1px solid var(--line)">'.join(
                        evidence_html(item, backend, show_title=True) for item in insufficient),
                    unsafe_allow_html=True,
                )
    elif evidence.get("evidence"):
        for item in evidence["evidence"]:
            with st.expander("NOT ASSESSED · {} · sentence {}".format(item["title"], item["sentence_id"])):
                st.markdown(
                    '<div class="ea-kicker">Retrieved Wikipedia sentence (not assessed)</div>'
                    '<div class="ea-serif">“{}”</div><div class="ea-sub">{}</div>'.format(
                        esc(item["text"]), source_line(dict(item, sentence=item["text"]), backend)),
                    unsafe_allow_html=True)

    freshness = ("The text is Wikipedia's current content and may change or be wrong."
                 if backend == "wikipedia_api" else
                 "The local corpus is a 2017 Wikipedia snapshot, so recent events are usually not covered.")
    st.caption(
        "Labels are LLM assessments of retrieved sentences, not independently verified ground truth. "
        "Retrieved pages may be off-topic. " + freshness
    )


def render_interpretation(interp):
    section(7, "Interpretation")
    review = ""
    if interp["human_review"]:
        title = "Human review strongly recommended" if interp["review_level"] == "strong" \
            else "Human review recommended"
        review = '<div class="ea-kicker" style="margin-top:14px">{}</div><ul>{}</ul>'.format(
            esc(title), "".join("<li>{}</li>".format(esc(r)) for r in interp["review_reasons"]))
    st.markdown(
        '<div class="ea-card ea-verdict {}"><h3>{}</h3><ul>{}</ul>{}</div>'.format(
            esc(interp["tone"]), esc(interp["headline"]),
            "".join("<li>{}</li>".format(esc(p)) for p in interp["points"]), review),
        unsafe_allow_html=True,
    )
    st.caption("This interpretation reports how independent signals relate to each other. "
               "It is not a fact-checking verdict and no combined score is computed.")


def render_technical(report):
    section(8, "Technical details")
    with st.expander("Component status and timings"):
        st.dataframe(
            [{"Component": name, "Status": c["status"],
              "Seconds": c.get("seconds"), "Message": c.get("message") or ""}
             for name, c in report["components"].items()],
            hide_index=True, width="stretch",
        )
    with st.expander("Raw analysis output (JSON)"):
        st.json({k: v for k, v in report.items() if k != "input"} | {
            "input": {k: v for k, v in report["input"].items() if k != "text"}})


def main():
    warm_models()
    status = cached_status()
    render_header(status)
    submitted, headline, body = render_input()

    if submitted:
        key = hashlib.sha256((headline + "\x00" + body).encode("utf-8")).hexdigest()
        if st.session_state.get("report_key") != key:
            try:
                with st.status("Analyzing…", expanded=False) as box:
                    report = run_analysis(body, headline, progress=lambda m: box.update(label=m + "…"))
                    box.update(label="Analysis complete", state="complete")
                st.session_state["report"] = report
                st.session_state["report_key"] = key
            except InputError as exc:
                st.warning(str(exc))
                st.session_state.pop("report", None)
                st.session_state.pop("report_key", None)
            except Exception:
                logger.exception("Analysis failed")
                st.error("The analysis failed unexpectedly. Please try again.")

    report = st.session_state.get("report")
    if report:
        render_classifier(report["classifier"])
        render_article(report)
        render_event(report["event"], report["components"]["event"])
        render_llm(report["llm"], report["components"]["llm"])
        render_evidence(report["evidence"], report["components"]["evidence"])
        render_interpretation(report["interpretation"])
        render_technical(report)

    st.markdown(
        '<div class="ea-foot">EvidenceAware · research prototype. Outputs are model predictions and '
        'LLM assessments, not fact-checking verdicts. Do not use as the sole basis for decisions.</div>',
        unsafe_allow_html=True,
    )


main()
