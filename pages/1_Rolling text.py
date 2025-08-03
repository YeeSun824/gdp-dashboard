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
        # extract everything between quotes
        import re
        tokens = re.findall(r'"([^"]+)"', text)
        return [t.strip() for t in tokens if t.strip()]
    # fallback ‑ split on commas
    return [t.strip() for t in text.split(',') if t.strip()]


def classify_rows(df: pd.DataFrame, text_col: str, dictionary: List[str]) -> pd.DataFrame:
    """Add prediction & matched keyword info to dataframe."""
    lower_dict = [k.lower() for k in dictionary]
    def _match_keywords(text: str):
        text_l = str(text).lower()
        return [kw for kw in lower_dict if kw in text_l]

    df = df.copy()
    df["_matched_keywords"] = df[text_col].apply(_match_keywords)
    df["predicted"] = df["_matched_keywords"].apply(lambda lst: 1 if lst else 0)
    return df


def compute_metrics(df: pd.DataFrame, truth_col: str) -> Dict[str, float]:
    TP = ((df["predicted"] == 1) & (df[truth_col] == 1)).sum()
    FP = ((df["predicted"] == 1) & (df[truth_col] == 0)).sum()
    FN = ((df["predicted"] == 0) & (df[truth_col] == 1)).sum()
    TN = ((df["predicted"] == 0) & (df[truth_col] == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if len(df) else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy
    }


def keyword_impact(df: pd.DataFrame, truth_col: str, text_col: str, dictionary: List[str]):
    """Return per‑keyword precision/recall/F1 stats sorted lists."""
    metrics = []
    for kw in dictionary:
        pred_pos = df[df[text_col].str.contains(kw, case=False, na=False)]
        TP = ((pred_pos[truth_col] == 1)).sum()
        FP = ((pred_pos[truth_col] == 0)).sum()
        total_pos_truth = (df[truth_col] == 1).sum()
        recall = TP / total_pos_truth if total_pos_truth else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0
        metrics.append({
            "keyword": kw,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "TP": TP,
            "FP": FP,
        })
    df_kw = pd.DataFrame(metrics)
    by_recall = df_kw.sort_values("recall", ascending=False)
    by_precision = df_kw.sort_values("precision", ascending=False)
    by_f1 = df_kw.sort_values("f1", ascending=False)
    return by_recall, by_precision, by_f1


# ---------------------------------------------------------------------
# Sidebar – Upload & Setup
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
    csv_bytes = uploaded.read()
    csv_text = csv_bytes.decode('utf‑8')
else:
    if st.sidebar.checkbox("Use sample data", value=True):
        csv_text = sample_csv
    else:
        csv_text = ""

if csv_text:
    df = pd.read_csv(StringIO(csv_text))
    st.sidebar.success(f"Loaded {len(df)} rows.")
    columns = df.columns.tolist()
else:
    df = pd.DataFrame()
    columns = []

text_col = st.sidebar.selectbox("Text column", options=columns, index=columns.index("Statement") if "Statement" in columns else 0 if columns else 0)
truth_col = st.sidebar.selectbox("Ground‑truth column (optional)", options=["(none)"]+columns, index=columns.index("Answer")+1 if "Answer" in columns else 0)
truth_col = None if truth_col == "(none)" else truth_col

st.sidebar.markdown("---")

st.sidebar.subheader("Keyword Dictionary")
if "dictionary_text" not in st.session_state:
    st.session_state.dictionary_text = "custom, customized, customization"

dictionary_text = st.sidebar.text_area("Enter keywords", value=st.session_state.dictionary_text, height=120)
dictionary = parse_dictionary(dictionary_text)
st.sidebar.info(f"{len(dictionary)} keywords loaded")

st.session_state.dictionary_text = dictionary_text  # persist

# ---------------------------------------------------------------------
# Main – Classification & Results
# ---------------------------------------------------------------------

st.title("Dictionary Classification Bot")

if not df.empty and dictionary:
    with st.spinner("Classifying ..."):
        df_pred = classify_rows(df, text_col, dictionary)
        st.subheader("Dataset with Predictions")
        st.dataframe(df_pred[[text_col, "predicted", "_matched_keywords"] + ([truth_col] if truth_col else [])])

        if truth_col:
            metrics = compute_metrics(df_pred, truth_col)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
            col4.metric("F1", f"{metrics['f1']*100:.2f}%")

            with st.expander("Confusion Matrix details"):
                st.write(metrics)

            # False Positives & Negatives
            fp_df = df_pred[(df_pred['predicted']==1) & (df_pred[truth_col]==0)]
            fn_df = df_pred[(df_pred['predicted']==0) & (df_pred[truth_col]==1)]
            st.subheader("Error Analysis")
            with st.expander(f"False Positives ({len(fp_df)})"):
                st.dataframe(fp_df[[text_col, "_matched_keywords"]])
            with st.expander(f"False Negatives ({len(fn_df)})"):
                st.dataframe(fn_df[[text_col]])

            # Keyword Impact
            st.markdown("---")
            st.subheader("Keyword Impact Analysis")
            by_rec, by_prec, by_f1 = keyword_impact(df_pred, truth_col, text_col, dictionary)
            tab1, tab2, tab3 = st.tabs(["Top by Recall", "Top by Precision", "Top by F1"])
            tab1.dataframe(by_rec.head(10))
            tab2.dataframe(by_prec.head(10))
            tab3.dataframe(by_f1.head(10))

            # Download CSV of keyword impact
            csv_kw = by_f1.to_csv(index=False).encode("utf‑8")
            st.download_button("Download keyword impact (top by F1)", csv_kw, file_name="keyword_impact.csv", mime="text/csv")
else:
    st.info("➡️ Upload a CSV and enter at least one keyword to begin.")
