import streamlit as st
import pandas as pd
from io import StringIO
from typing import List, Dict

st.set_page_config(page_title="Dictionary Classification Bot", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def parse_dictionary(text: str) -> List[str]:
    """Parse user-supplied dictionary text into a list of keywords."""
    if not text:
        return []
    import re
    # Supports both comma‑separated and "quoted" list formats
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return [t.strip() for t in quoted if t.strip()]
    return [t.strip() for t in text.split(',') if t.strip()]


def classify_rows(df: pd.DataFrame, text_col: str, dictionary: List[str]) -> pd.DataFrame:
    """Add prediction & matched keyword info to dataframe."""
    lowered = [k.lower() for k in dictionary]

    def _match(txt: str):
        txt_l = str(txt).lower()
        return [kw for kw in lowered if kw in txt_l]

    df = df.copy()
    df["_matched_keywords"] = df[text_col].apply(_match)
    df["predicted"] = df["_matched_keywords"].apply(lambda x: 1 if x else 0)
    return df


def compute_metrics(df: pd.DataFrame, truth_col: str) -> Dict[str, float]:
    TP = ((df['predicted'] == 1) & (df[truth_col] == 1)).sum()
    FP = ((df['predicted'] == 1) & (df[truth_col] == 0)).sum()
    FN = ((df['predicted'] == 0) & (df[truth_col] == 1)).sum()
    TN = ((df['predicted'] == 0) & (df[truth_col] == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if len(df) else 0

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def keyword_impact(df: pd.DataFrame, truth_col: str, text_col: str, dictionary: List[str]):
    """Return per‑keyword precision/recall/F1 stats sorted lists."""
    stats = []
    total_pos = (df[truth_col] == 1).sum()

    for kw in dictionary:
        mask_kw = df[text_col].str.contains(kw, case=False, na=False)
        tp = ((mask_kw) & (df[truth_col] == 1)).sum()
        fp = ((mask_kw) & (df[truth_col] == 0)).sum()
        rec = tp / total_pos if total_pos else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else 0
        stats.append({"keyword": kw, "recall": rec, "precision": prec, "f1": f1, "TP": tp, "FP": fp})

    df_kw = pd.DataFrame(stats)
    return (df_kw.sort_values('recall', ascending=False),
            df_kw.sort_values('precision', ascending=False),
            df_kw.sort_values('f1', ascending=False))

# ---------------------------------------------------------------------
# Sidebar – Upload & Configuration
# ---------------------------------------------------------------------

st.sidebar.title("🗂️ Upload CSV & Configure")

SAMPLE_CSV = """ID,Statement,Answer
1,It's SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want (, , , , etc) & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some \"wardrobe crunches,\" and check your basics! Never on sale.,1
5,He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.,0"""

upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if upload is not None:
    csv_text = upload.getvalue().decode("utf-8", errors="ignore")
elif st.sidebar.checkbox("Use sample data", value=True):
    csv_text = SAMPLE_CSV
else:
    csv_text = ""

if csv_text:
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read CSV: {e}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

columns = df.columns.tolist()

# Text column selectbox only if we have columns
if columns:
    default_text_idx = columns.index("Statement") if "Statement" in columns else 0
    text_col = st.sidebar.selectbox("Text column", options=columns, index=default_text_idx)
else:
    text_col = None

truth_options = ["(none)"] + columns
if columns:
    default_truth_idx = truth_options.index("Answer") if "Answer" in truth_options else 0
else:
    default_truth_idx = 0
truth_sel = st.sidebar.selectbox("Ground‑truth column (optional)", options=truth_options, index=default_truth_idx)
truth_col = None if truth_sel == "(none)" else truth_sel

st.sidebar.markdown("---")

st.sidebar.subheader("Keyword Dictionary")
if "dict_text" not in st.session_state:
    st.session_state.dict_text = "custom, customized, customization"

st.session_state.dict_text = st.sidebar.text_area("Enter keywords", value=st.session_state.dict_text, height=120)
dictionary = parse_dictionary(st.session_state.dict_text)
st.sidebar.info(f"{len(dictionary)} keywords loaded")

# ---------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------

st.title("Dictionary Classification Bot")

if df.empty:
    st.info("➡️ Upload a CSV (or enable sample data) to begin.")
    st.stop()

if not dictionary:
    st.info("➡️ Please enter at least one keyword.")
    st.stop()

if text_col is None:
    st.error("No columns detected – please check your CSV.")
    st.stop()

with st.spinner("Classifying …"):
    df_pred = classify_rows(df, text_col, dictionary)

st.subheader("Dataset with Predictions")
preview_cols = [text_col, "predicted", "_matched_keywords"] + ([truth_col] if truth_col else [])
st.dataframe(df_pred[preview_cols])

if truth_col:
    metrics = compute_metrics(df_pred, truth_col)
    met_cols = st.columns(4)
    met_cols[0].metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    met_cols[1].metric("Precision", f"{metrics['precision']*100:.2f}%")
    met_cols[2].metric("Recall", f"{metrics['recall']*100:.2f}%")
    met_cols[3].metric("F1", f"{metrics['f1']*100:.2f}%")

    fp_df = df_pred[(df_pred['predicted'] == 1) & (df_pred[truth_col] == 0)]
    fn_df = df_pred[(df_pred['predicted'] == 0) & (df_pred[truth_col] == 1)]

    st.subheader("Error Analysis")
    with st.expander(f"False Positives ({len(fp_df)})"):
        st.dataframe(fp_df[[text_col, "_matched_keywords"]])
    with st.expander(f"False Negatives ({len(fn_df)})"):
        st.dataframe(fn_df[[text_col]])

    st.markdown("---")
    st.subheader("Keyword Impact Analysis")
    rec, prec, f1_tbl = keyword_impact(df_pred, truth_col, text_col, dictionary)
    tabs = st.tabs(["Top by Recall", "Top by Precision", "Top by F1"])
    tabs[0].dataframe(rec.head(10))
    tabs[1].dataframe(prec.head(10))
    tabs[2].dataframe(f1_tbl.head(10))

    csv_bytes = f1_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("Download keyword impact (top by F1)", csv_bytes, "keyword_impact.csv", "text/csv")
