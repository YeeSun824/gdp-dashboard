import streamlit as st
import pandas as pd
import nltk
import re
from io import StringIO

# Optional third‑party libraries ------------------------------------------------
try:
    import emoji  # 👉 pip install emoji
except ModuleNotFoundError:
    st.error("Missing dependency: `emoji`. Run `pip install emoji` and restart the app.")
    st.stop()

try:
    from unidecode import unidecode  # 👉 pip install Unidecode
except ModuleNotFoundError:
    st.error("Missing dependency: `Unidecode`. Run `pip install Unidecode` and restart the app.")
    st.stop()

# ----------------------------------------------------------------------------
# NLTK setup (quiet download of punkt if not already cached)
# ----------------------------------------------------------------------------
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Regex patterns used for cleaning captions
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
EMAIL_RE = re.compile(r"\\S+@\\S+")
HASHTAG_RE = re.compile(r"#[\\w\\d]+")
MENTION_RE = re.compile(r"@[\\w\\d]+")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def clean_caption(
    text: str,
    *,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_hashtags: bool = True,
    remove_mentions: bool = True,
) -> str:
    """Apply the suite of cleaning operations from the revised script.

    Steps (in order):
      1. Remove emoji characters.
      2. Transliterate to ASCII with `unidecode`.
      3. Strip URLs / emails / hashtags / mentions (configurable via flags).
      4. Collapse newlines + excess spaces.
      5. Fallback token "[PAD]" if resulting string is empty.
    """
    if not isinstance(text, str):
        return "[PAD]"

    # 1️⃣ Remove emoji (fast and reliable)
    text = emoji.replace_emoji(text, replace="")

    # 2️⃣ Normalise accents / curly quotes, etc.
    text = unidecode(text)

    # 3️⃣ Regex‑based removals
    if remove_urls:
        text = URL_RE.sub("", text)
    if remove_emails:
        text = EMAIL_RE.sub("", text)
    if remove_hashtags:
        text = HASHTAG_RE.sub("", text)
    if remove_mentions:
        text = MENTION_RE.sub("", text)

    # 4️⃣ Tidy whitespace
    text = text.replace("\n", " ")
    text = " ".join(text.split()).strip()

    return text if text else "[PAD]"


def split_sentences(caption: str):
    """Sentence‑tokenise while ensuring a trailing punctuation mark if absent."""
    if caption and caption[-1] not in ".!?":
        caption += "."
    return [s.strip() for s in nltk.sent_tokenize(caption) if s.strip()]


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Instagram Caption Transformer", page_icon="✨", layout="centered")
st.title("Instagram Caption Transformer ✨")
st.write(
    "Upload a CSV, clean each caption, **explode** captions into sentences, and download the transformed dataset "
    "with the exact column ordering: `shortcode`, `turn`, `caption`, `transcript`, `post_url`."
)

# ---- Sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("CSV file", type=["csv"], help="Raw Instagram data with at least a caption column.")

    caption_col_name = st.text_input("Caption column", value="caption")

    id_col_name = st.text_input("Shortcode / post_id column", value="shortcode")

    post_url_handling = st.selectbox(
        "Post URL source",
        options=["Construct from shortcode", "Use column"],
        index=0,
    )
    if post_url_handling == "Use column":
        post_url_col_name = st.text_input("Post URL column", value="post_url")
    else:
        post_url_col_name = None  # constructed later

    st.markdown("---")
    st.subheader("Cleaning options")
    remove_urls = st.checkbox("Remove URLs", value=True)
    remove_emails = st.checkbox("Remove email addresses", value=True)
    remove_hashtags = st.checkbox("Remove #hashtags", value=True)
    remove_mentions = st.checkbox("Remove @mentions", value=True)

    transform_btn = st.button("Transform 🚀", disabled=uploaded_file is None)

# ---- Main panel ------------------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Upload a CSV file in the sidebar to get started.")
    st.stop()

# Read CSV (auto‑detect encoding; Streamlit reads bytes)
try:
    df_raw = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
    # Fallback encodings
    for enc in ["latin1", "iso-8859-1"]:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("Failed to read CSV with common encodings (utf‑8, latin1, iso‑8859‑1).")
        st.stop()

st.subheader("Raw data preview")
st.dataframe(df_raw.head())

if not transform_btn:
    st.stop()

# Validate required columns ---------------------------------------------------
for col in [caption_col_name, id_col_name]:
    if col not in df_raw.columns:
        st.error(f"Required column '{col}' not found in uploaded data.")
        st.stop()

if post_url_col_name and post_url_col_name not in df_raw.columns:
    st.warning(f"Column '{post_url_col_name}' not found – post_url will be empty.")

# Clean captions -------------------------------------------------------------
st.info("Cleaning captions … this may take a few seconds for large files.")

clean_flags = dict(
    remove_urls=remove_urls,
    remove_emails=remove_emails,
    remove_hashtags=remove_hashtags,
    remove_mentions=remove_mentions,
)

# Apply cleaning row‑wise
cleaned = df_raw[caption_col_name].apply(lambda x: clean_caption(x, **clean_flags))

# Build transformed records --------------------------------------------------
records = []
for idx, row in df_raw.iterrows():
    caption_orig = row[caption_col_name]
    caption_clean = cleaned.iloc[idx]

    # Sentence splitting
    sentences = split_sentences(caption_clean)

    # Shortcode / ID
    shortcode_val = str(row[id_col_name]) if pd.notna(row[id_col_name]) else ""

    # Post URL
    if post_url_col_name:
        post_url_val = str(row[post_url_col_name]) if post_url_col_name in row and pd.notna(row[post_url_col_name]) else ""
    else:
        post_url_val = f"https://www.instagram.com/p/{shortcode_val}/" if shortcode_val else ""

    for turn, sent in enumerate(sentences, 1):
        records.append({
            "shortcode": shortcode_val,
            "turn": turn,
            "caption": caption_orig,
            "transcript": sent,
            "post_url": post_url_val,
        })

if not records:
    st.error("No sentences extracted – please check your caption column or cleaning options.")
    st.stop()

# Final DataFrame ------------------------------------------------------------
df_out = pd.DataFrame(records, columns=["shortcode", "turn", "caption", "transcript", "post_url"])

st.success(f"Processed {len(df_out)} sentences from {len(df_raw)} captions ✔️")

st.subheader("Transformed data preview")
st.dataframe(df_out.head())

# Download button ------------------------------------------------------------
csv_buffer = StringIO()
df_out.to_csv(csv_buffer, index=False, quoting=1, quotechar='"', escapechar='\\')  # QUOTE_NONNUMERIC = 1
st.download_button(
    label="Download transformed CSV",
    data=csv_buffer.getvalue(),
    file_name="ig_posts_transformed.csv",
    mime="text/csv",
)

st.caption(
    "Built with Streamlit · Implements the full cleaning & transformation pipeline from the revised script by Dr. Yufan (Frank) Lin."
)
