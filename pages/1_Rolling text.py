import streamlit as st
import pandas as pd
from io import StringIO
from typing import List, Dict

st.set_page_config(
    page_title="Dictionary Classification Bot",
    page_icon="🔍",
    layout="wide",
)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text))


def parse_dictionary(text: str) -> List[str]:
    """Return a list of keywords from user text (quoted or comma‑sep)."""
    import re
    if not text:
        return []
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return [k.strip() for k in quoted if k.strip()]
    return [k.strip() for k in text.split(',') if k.strip()]


def classify(df: pd.DataFrame, text_col: str, keywords: List[str]):
    lowered = [k.lower() for k in keywords]

    def _match(t: str):
        tl = str(t).lower()
        return [kw for kw in lowered if kw in tl]

    res = df.copy()
    res["_matched_keywords"] = res[text_col].apply(_match)
    res["predicted"] = res["_matched_keywords"].apply(lambda lst: 1 if lst else 0)
    return res


def metrics(df: pd.DataFrame, truth_col: str):
    TP = ((df.predicted == 1) & (df[truth_col] == 1)).sum()
    FP = ((df.predicted == 1) & (df[truth_col] == 0)).sum()
    FN = ((df.predicted == 0) & (df[truth_col] == 1)).sum()
    TN = ((df.predicted == 0) & (df[truth_col] == 0)).sum()
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if len(df) else 0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec, f1=f1, accuracy=acc)


def keyword_stats(df: pd.DataFrame, truth_col: str, text_col: str, keywords: List[str]):
    pos_total = (df[truth_col] == 1).sum()
    data = []
    for kw in keywords:
        mask = df[text_col].str.contains(kw, case=False, na=False)
        tp = ((mask) & (df[truth_col] == 1)).sum()
        fp = ((mask) & (df[truth_col] == 0)).sum()
        rec = tp / pos_total if pos_total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2*prec*rec/(prec+rec) if prec and rec else 0
        data.append(dict(keyword=kw, recall=rec, precision=prec, f1=f1, TP=tp, FP=fp))
    kw_df = pd.DataFrame(data)
    return kw_df

# ------------------------  Session Defaults  -----------------------

if "csv_text" not in st.session_state:
    st.session_state.csv_text = ""
if "classified" not in st.session_state:
    st.session_state.classified = False
if "dict_text" not in st.session_state:
    st.session_state.dict_text = "custom, customized, customization"

# -----------------------------  UI  --------------------------------

st.title("🔍 Dictionary Classification Bot")

step_tabs = st.tabs(["1️⃣  Upload Data", "2️⃣  Keywords", "3️⃣  Results"])

# --------------------------------------------------  Tab 1 – Upload
with step_tabs[0]:
    st.header("Step 1 — Provide a CSV dataset")
    col_upl, col_opts = st.columns([3, 2])

    with col_upl:
        upload = st.file_uploader("Drag‑and‑drop or click to upload", type="csv")
        sample = st.checkbox("Use built‑in sample", value=not upload)

    SAMPLE_CSV = (
        """ID,Statement,Answer\n"
        "1,It's SPRING TRUNK SHOW week!,1\n"
        "2,I am offering 4 shirts styled the way you want (, , , , etc) & the 5th is ... MAGNETIC COLLAR STAY ...,1\n"
        "3,In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!,0"""
    )

    if upload:
        st.session_state.csv_text = upload.getvalue().decode("utf‑8", errors="ignore")
    elif sample:
        st.session_state.csv_text = SAMPLE_CSV

    if st.session_state.csv_text:
        try:
            df_raw = load_csv(st.session_state.csv_text)
            st.success(f"Loaded {len(df_raw)} rows · {len(df_raw.columns)} columns")
            with st.expander("Preview first 5 rows"):
                st.dataframe(df_raw.head())
        except Exception as e:
            st.error(f"❌ Could not parse CSV: {e}")
            df_raw = pd.DataFrame()
    else:
        st.info("⬅️ Upload a file or tick ‘Use built‑in sample’ to continue")
        df_raw = pd.DataFrame()

    # Column selections
    if not df_raw.empty:
        text_col = st.selectbox("Which column contains the text?", df_raw.columns, 0, key="text_col")
        truth_col_opt = st.selectbox("Ground‑truth column (optional)", ["(none)"]+df_raw.columns.tolist(), key="truth_col")
        truth_col = None if truth_col_opt == "(none)" else truth_col_opt
        st.session_state.df_raw = df_raw
        st.session_state.text_col = text_col
        st.session_state.truth_col = truth_col

# -------------------------------------------  Tab 2 – Dictionary
with step_tabs[1]:
    st.header("Step 2 — Enter keyword dictionary")

    st.session_state.dict_text = st.text_area(
        "Comma‑separated list or \"quoted\" list of keywords",
        value=st.session_state.dict_text,
        height=120,
    )

    keywords = parse_dictionary(st.session_state.dict_text)
    st.markdown(f"**{len(keywords)}** keyword(s) loaded")

    if keywords:
        st.write("### Parsed keywords")
        st.dataframe(pd.DataFrame({"Keyword": keywords}))

    st.info("Proceed to **Results** once data & keywords are ready →")

# ----------------------------------------------  Tab 3 – Results
with step_tabs[2]:
    st.header("Step 3 — Classification results & insights")

    # Preconditions
    if "df_raw" not in st.session_state or st.session_state.df_raw.empty:
        st.warning("⬅️ Please upload data first (Step 1)")
        st.stop()
    if not keywords:
        st.warning("⬅️ Please enter at least one keyword (Step 2)")
        st.stop()

    # Run / rerun classification only on demand
    if st.button("🔎  Run classification", key="run_btn") or not st.session_state.get("classified", False):
        with st.spinner("Classifying ..."):
            st.session_state.df_pred = classify(
                st.session_state.df_raw, st.session_state.text_col, keywords
            )
            st.session_state.classified = True
            st.success("Done!")

    if not st.session_state.classified:
        st.info("Click the **Run classification** button")
        st.stop()

    df_pred = st.session_state.df_pred
    truth_col = st.session_state.truth_col

    st.subheader("Predictions")
    show_cols = [st.session_state.text_col, "predicted", "_matched_keywords"]
    if truth_col:
        show_cols.append(truth_col)
    st.dataframe(df_pred[show_cols])

    # Download predictions
    csv_pred = df_pred.to_csv(index=False).encode("utf‑8")
    st.download_button("⬇️  Download predictions CSV", csv_pred, "predictions.csv", "text/csv")

    # Metrics & error inspection
    if truth_col:
        m = metrics(df_pred, truth_col)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        metric_cols[1].metric("Precision", f"{m['precision']*100:.2f}%")
        metric_cols[2].metric("Recall", f"{m['recall']*100:.2f}%")
        metric_cols[3].metric("F1", f"{m['f1']*100:.2f}%")

        fp = df_pred[(df_pred.predicted == 1) & (df_pred[truth_col] == 0)]
        fn = df_pred[(df_pred.predicted == 0) & (df_pred[truth_col] == 1)]

        with st.expander(f"🔴  False Positives ({len(fp)})"):
            st.dataframe(fp[[st.session_state.text_col, "_matched_keywords"]])
        with st.expander(f"🟡  False Negatives ({len(fn)})"):
            st.dataframe(fn[[st.session_state.text_col]])

        st.markdown("---")
        st.subheader("Keyword impact analysis")
        kw_df = keyword_stats(df_pred, truth_col, st.session_state.text_col, keywords)
        kw_tabs = st.tabs(["Recall", "Precision", "F1"])
        kw_tabs[0].dataframe(kw_df.sort_values("recall", ascending=False).head(15))
        kw_tabs[1].dataframe(kw_df.sort_values("precision", ascending=False).head(15))
        kw_tabs[2].dataframe(kw_df.sort_values("f1", ascending=False).head(15))

        csv_kw = kw_df.to_csv(index=False).encode("utf‑8")
        st.download_button("⬇️  Download keyword impact", csv_kw, "keyword_impact.csv", "text/csv")
