import streamlit as st
import pandas as pd
from io import StringIO
from typing import List, Dict

st.set_page_config(page_title="Dictionary Classification Bot", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def parse_dictionary(text: str) -> List[str]:
    """Parse user‑supplied dictionary text into a list of keywords."""
    if not text:
        return []
    # Support both comma‑separated and "quoted" formats
    if '"' in text:
        import re
        tokens = re.findall(r'"([^\"]+)"', text)
        return [t.strip() for t in tokens if t.strip()]
    return [t.strip() for t in text.split(',') if t.strip()]


def classify_rows(df: pd.DataFrame, text_col: str, dictionary: List[str]) -> pd.DataFrame:
    """Add prediction & matched‑keyword info to the DataFrame."""
    lowered = [k.lower() for k in dictionary]

    def _match(txt: str):
        txt_l = str(txt).lower()
        return [kw for kw in lowered if kw in txt_l]

    out = df.copy()
    out["_matched_keywords"] = out[text_col].apply(_match)
    out["predicted"] = out["_matched_keywords"].apply(lambda lst: 1 if lst else 0)
    return out


def compute_metrics(df: pd.DataFrame, truth_col: str) -> Dict[str, float]:
    TP = ((df["predicted"] == 1) & (df[truth_col] == 1)).sum()
    FP = ((df["predicted"] == 1) & (df[truth_col] == 0)).sum()
    FN = ((df["predicted"] == 0) & (df[truth_col] == 1)).sum()
    TN = ((df["predicted"] == 0) & (df[truth_col] == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if len(df) else 0

    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def keyword_impact(df: pd.DataFrame, truth_col: str, text_col: str, dictionary: List[str]):
    """Return three DataFrames ranked by recall, precision, and F1."""
    metrics = []
    total_pos = (df[truth_col] == 1).sum()

    for kw in dictionary:
        mask = df[text_col].str.contains(kw, case=False, na=False)
        TP = ((mask) & (df[truth_col] == 1)).sum()
        FP = ((mask) & (df[truth_col] == 0)).sum()
        recall = TP / total_pos if total_pos else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0
        metrics.append(dict(keyword=kw, recall=recall, precision=precision, f1=f1, TP=TP, FP=FP))

    kw_df = pd.DataFrame(metrics)
    return (kw_df.sort_values("recall", ascending=False),
            kw_df.sort_values("precision", ascending=False),
            kw_df.sort_values("f1", ascending=False))

# ---------------------------------------------------------------------
# Sidebar – Upload & Configuration
# ---------------------------------------------------------------------

st.sidebar.title("🗂️ Upload CSV & Configure")

sample_csv = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want (, , , , etc) & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some \"wardrobe crunches,\" and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.,0"""

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    csv_text = uploaded.getvalue().decode("utf-8", errors="ignore")
else:
    csv_text = sample_csv if st.sidebar.checkbox("Use sample data", True) else ""

if csv_text:
    try:
        df = pd.read_csv(StringIO(csv_text))
        st.sidebar.success(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to parse CSV: {e}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

cols = df.columns.tolist()
text_col = st.sidebar.selectbox("Text column", cols, index=cols.index("Statement") if "Statement" in cols else 0) if cols else None
truth_opt = st.sidebar.selectbox("Ground‑truth column (optional)", ["(none)"]+cols, index=(cols.index("Answer")+1) if "Answer" in cols else 0) if cols else "(none)"
truth_col = None if truth_opt == "(none)" else truth_opt

st.sidebar.markdown("---")

st.sidebar.subheader("Keyword Dictionary")
if "dict_text" not in st.session_state:
    st.session_state.dict_text = "custom, customized, customization"

st.session_state.dict_text = st.sidebar.text_area("Enter keywords", value=st.session_state.dict_text, height=120)
dictionary = parse_dictionary(st.session_state.dict_text)
st.sidebar.info(f"{len(dictionary)} keywords loaded")

# ---------------------------------------------------------------------
# Main – Classification & Results
# ---------------------------------------------------------------------

st.title("Dictionary Classification Bot")

if df.empty or not dictionary or text_col is None:
    st.info("➡️ Upload a CSV and enter at least one keyword to begin.")
    st.stop()

with st.spinner("Classifying …"):
    df_pred = classify_rows(df, text_col, dictionary)

st.subheader("Dataset with Predictions")
show_cols = [text_col, "predicted", "_matched_keywords"] + ([truth_col] if truth_col else [])
st.dataframe(df_pred[show_cols])

if truth_col:
    metrics = compute_metrics(df_pred, truth_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    c4.metric("F1", f"{metrics['f1']*100:.2f}%")

    fp = df_pred[(df_pred['predicted']==1) & (df_pred[truth_col]==0)]
    fn = df_pred[(df_pred['predicted']==0) & (df_pred[truth_col]==1)]

    st.subheader("Error Analysis")
    with st.expander(f"False Positives ({len(fp)})"):
        st.dataframe(fp[[text_col, "_matched_keywords"]])
    with st.expander(f"False Negatives ({len(fn)})"):
        st.dataframe(fn[[text_col]])

    st.markdown("---")
    st.subheader("Keyword Impact Analysis")
    by_rec, by_prec, by_f1 = keyword_impact(df_pred, truth_col, text_col, dictionary)
    t1, t2, t3 = st.tabs(["Top by Recall", "Top by Precision", "Top by F1"])
    t1.dataframe(by_rec.head(10))
    t2.dataframe(by_prec.head(10))
    t3.dataframe(by_f1.head(10))

    csv_kw = by_f1.to_csv(index=False).encode()
    st.download_button("Download keyword impact (top by F1)", csv_kw, "keyword_impact.csv", "text/csv")
