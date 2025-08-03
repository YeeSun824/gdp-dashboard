import streamlit as st
import pandas as pd
from io import StringIO
from typing import List
import textwrap

st.set_page_config(page_title="Dictionary Classification Bot", page_icon="🔍", layout="wide")

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(text: str) -> pd.DataFrame:
    """Try to read CSV with pandas; fall back to python engine if needed."""
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        # More tolerant parsing (python engine)
        return pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")


def parse_dictionary(text: str) -> List[str]:
    import re
    if not text:
        return []
    quoted = re.findall(r'"([^"]+)"', text)
    return [k.strip() for k in quoted] if quoted else [k.strip() for k in text.split(',') if k.strip()]


def classify(df: pd.DataFrame, text_col: str, keywords: List[str]):
    lowered = [k.lower() for k in keywords]

    def _match(t):
        tl = str(t).lower()
        return [kw for kw in lowered if kw in tl]

    out = df.copy()
    out["_matched_keywords"] = out[text_col].apply(_match)
    out["predicted"] = out["_matched_keywords"].apply(lambda lst: 1 if lst else 0)
    return out


def calc_metrics(df: pd.DataFrame, truth_col: str):
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
    rows = []
    for kw in keywords:
        mask = df[text_col].str.contains(kw, case=False, na=False)
        tp = ((mask) & (df[truth_col] == 1)).sum()
        fp = ((mask) & (df[truth_col] == 0)).sum()
        rec = tp / pos_total if pos_total else 0
        prec = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0
        rows.append(dict(keyword=kw, recall=rec, precision=prec, f1=f1, TP=tp, FP=fp))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Session Defaults
# ------------------------------------------------------------------

st.session_state.setdefault("csv_text", "")
st.session_state.setdefault("classified", False)
st.session_state.setdefault("dict_text", "custom, customized, customization")

# ------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------

st.title("🔍 Dictionary Classification Bot")

steps = st.tabs(["1️⃣ Upload", "2️⃣ Keywords", "3️⃣ Results"])

# -----------------------------  Step 1 – Upload
with steps[0]:
    st.header("Step 1 — Upload CSV or try the sample")
    col_upl, _ = st.columns([3, 2])

    with col_upl:
        file_obj = st.file_uploader("CSV file", type="csv")
        use_sample = st.checkbox("Use sample dataset", value=not file_obj)

    SAMPLE_CSV = textwrap.dedent(
        """ID,Statement,Answer
1,"It's SPRING TRUNK SHOW week!",1
2,"I am offering 4 shirts styled the way you want (, , , , etc) & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!",1
3,"In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!",0
4,"It is now time to do some 'wardrobe crunches,' and check your basics! Never on sale.",1
5,"He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.",0"""
    )

    if file_obj:
        st.session_state.csv_text = file_obj.read().decode("utf-8", errors="ignore")
    elif use_sample:
        st.session_state.csv_text = SAMPLE_CSV

    if st.session_state.csv_text:
        try:
            df_raw = load_csv(st.session_state.csv_text)
            st.success(f"Loaded {len(df_raw)} rows · {len(df_raw.columns)} columns")
            st.dataframe(df_raw.head())
        except Exception as e:
            st.error(f"❌ CSV parse error: {e}")
            df_raw = pd.DataFrame()
    else:
        st.info("Upload a CSV or select the sample to continue.")
        df_raw = pd.DataFrame()

    if not df_raw.empty:
        st.selectbox("Text column", df_raw.columns, key="text_col")
        st.selectbox("Ground‑truth column (optional)", ["(none)"]+df_raw.columns.tolist(), key="truth_col")
        st.session_state.df_raw = df_raw

# -----------------------------  Step 2 – Keywords
with steps[1]:
    st.header("Step 2 — Keyword dictionary")
    st.session_state.dict_text = st.text_area("Enter keywords (comma‑separated or \"quoted\")", value=st.session_state.dict_text, height=110)
    keywords = parse_dictionary(st.session_state.dict_text)
    st.markdown(f"Loaded **{len(keywords)}** keyword(s).")
    if keywords:
        st.dataframe(pd.DataFrame({"Keyword": keywords}))
    st.info("Move to Results when ready →")

# -----------------------------  Step 3 – Results
with steps[2]:
    st.header("Step 3 — Results & analysis")

    if "df_raw" not in st.session_state or st.session_state.df_raw.empty:
        st.warning("⬅️ Need data first (Step 1)")
        st.stop()
    if not keywords:
        st.warning("⬅️ Enter at least one keyword (Step 2)")
        st.stop()

    text_col = st.session_state.text_col
    truth_sel = st.session_state.truth_col
    truth_col = None if truth_sel == "(none)" else truth_sel

    if st.button("🔎 Run classification") or not st.session_state.classified:
        with st.spinner("Processing …"):
            st.session_state.df_pred = classify(st.session_state.df_raw, text_col, keywords)
            st.session_state.classified = True
            st.success("Classification complete")

    if not st.session_state.classified:
        st.stop()

    df_pred = st.session_state.df_pred
    cols_to_show = [text_col, "predicted", "_matched_keywords"] + ([truth_col] if truth_col else [])
    st.subheader("Predictions")
    st.dataframe(df_pred[cols_to_show])

    st.download_button("⬇️ Download predictions", df_pred.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    # Metrics & keyword impact
    if truth_col:
        m = calc_metrics(df_pred, truth_col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{m['precision']*100:.2f}%")
        c3.metric("Recall", f"{m['recall']*100:.2f}%")
        c4.metric("F1", f"{m['f1']*100:.2f}%")

        fp = df_pred[(df_pred.predicted == 1) & (df_pred[truth_col] == 0)]
        fn = df_pred[(df_pred.predicted == 0) & (df_pred[truth_col] == 1)]
        with st.expander(f"🔴 False Positives ({len(fp)})"):
            st.dataframe(fp[[text_col, "_matched_keywords"]])
        with st.expander(f"🟡 False Negatives ({len(fn)})"):
            st.dataframe(fn[[text_col]])

        st.markdown("---")
        st.subheader("Keyword impact")
        kw_df = keyword_stats(df_pred, truth_col, text_col, keywords)
        tab_rec, tab_prec, tab_f1 = st.tabs(["Recall", "Precision", "F1"])
        tab_rec.dataframe(kw_df.sort_values("recall", ascending=False))
        tab_prec.dataframe(kw_df.sort_values("precision", ascending=False))
        tab_f1.dataframe(kw_df.sort_values("f1", ascending=False))

        st.download_button("⬇️ Download keyword impact", kw_df.to_csv(index=False).encode(), "keyword_impact.csv", "text/csv")
