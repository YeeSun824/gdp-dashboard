import streamlit as st
import pandas as pd
import json
from io import StringIO

# --- Page Configuration -----------------------------------------------------
st.set_page_config(
    page_title="Marketing Keywords Classifier",
    page_icon="🎯",
    layout="wide"
)

# --- Initialize Session State -----------------------------------------------
if 'dictionaries' not in st.session_state:
    st.session_state.dictionaries = {
        'urgency_marketing': {
            'limited', 'limited time', 'limited run', 'limited edition', 'order now',
            'last chance', 'hurry', 'while supplies last', "before they're gone",
            'selling out', 'selling fast', 'act now', "don't wait", 'today only',
            'expires soon', 'final hours', 'almost gone'
        },
        'exclusive_marketing': {
            'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
            'members only', 'vip', 'special access', 'invitation only',
            'premium', 'privileged', 'limited access', 'select customers',
            'insider', 'private sale', 'early access'
        }
    }

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# --- Helper Functions -------------------------------------------------------
def classify_statement(text: str, dictionaries: dict) -> list[str]:
    """Return a list of dictionary names whose keywords appear in *text*."""
    text_lower = str(text).lower()
    matched = []
    for label, keywords in dictionaries.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(label)
    return matched

def process_dataframe(df: pd.DataFrame, dictionaries: dict, text_column: str) -> pd.DataFrame:
    """Process the dataframe with classification."""
    df_copy = df.copy()
    
    # Classify each statement
    df_copy['labels'] = df_copy[text_column].astype(str).apply(
        lambda x: classify_statement(x, dictionaries)
    )
    
    # One-hot encode each category
    for label in dictionaries:
        df_copy[label] = df_copy['labels'].apply(lambda cats: label in cats)
    
    return df_copy

# --- Main App ---------------------------------------------------------------
st.title("🎯 Marketing Keywords Classifier")
st.markdown("Upload your dataset and classify text based on customizable marketing keyword dictionaries.")

# --- Sidebar for Dictionary Management -------------------------------------
st.sidebar.header("📚 Manage Dictionaries")

# Dictionary selection
dict_names = list(st.session_state.dictionaries.keys())
selected_dict = st.sidebar.selectbox("Select Dictionary", dict_names)

# Add new dictionary
st.sidebar.subheader("Add New Dictionary")
new_dict_name = st.sidebar.text_input("Dictionary Name")
if st.sidebar.button("Add Dictionary") and new_dict_name:
    if new_dict_name not in st.session_state.dictionaries:
        st.session_state.dictionaries[new_dict_name] = set()
        st.sidebar.success(f"Added dictionary: {new_dict_name}")
        st.rerun()
    else:
        st.sidebar.error("Dictionary already exists!")

# Edit selected dictionary
if selected_dict:
    st.sidebar.subheader(f"Edit: {selected_dict}")
    
    # Display current keywords
    current_keywords = st.session_state.dictionaries[selected_dict]
    keywords_text = '\n'.join(sorted(current_keywords))
    
    # Text area for editing keywords
    edited_keywords = st.sidebar.text_area(
        "Keywords (one per line)",
        value=keywords_text,
        height=200,
        key=f"keywords_{selected_dict}"
    )
    
    # Update keywords
    if st.sidebar.button("Update Keywords", key=f"update_{selected_dict}"):
        new_keywords = set(line.strip() for line in edited_keywords.split('\n') if line.strip())
        st.session_state.dictionaries[selected_dict] = new_keywords
        st.sidebar.success("Keywords updated!")
        st.rerun()
    
    # Delete dictionary
    if st.sidebar.button("Delete Dictionary", key=f"delete_{selected_dict}"):
        if len(st.session_state.dictionaries) > 1:
            del st.session_state.dictionaries[selected_dict]
            st.sidebar.success(f"Deleted dictionary: {selected_dict}")
            st.rerun()
        else:
            st.sidebar.error("Cannot delete the last dictionary!")

# Export/Import dictionaries
st.sidebar.subheader("Import/Export")
if st.sidebar.button("Export Dictionaries"):
    # Convert sets to lists for JSON serialization
    export_data = {k: list(v) for k, v in st.session_state.dictionaries.items()}
    st.sidebar.download_button(
        label="Download JSON",
        data=json.dumps(export_data, indent=2),
        file_name="dictionaries.json",
        mime="application/json"
    )

uploaded_dict = st.sidebar.file_uploader("Import Dictionaries", type=['json'])
if uploaded_dict is not None:
    try:
        import_data = json.load(uploaded_dict)
        # Convert lists back to sets
        imported_dicts = {k: set(v) for k, v in import_data.items()}
        st.session_state.dictionaries.update(imported_dicts)
        st.sidebar.success("Dictionaries imported successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error importing dictionaries: {str(e)}")

# --- Main Content -----------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset with {len(df)} rows and {len(df.columns)} columns")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if text_columns:
                selected_column = st.selectbox(
                    "Select text column to classify",
                    text_columns,
                    index=0 if 'Statement' in text_columns else 0
                )
                
                # Process button
                if st.button("🚀 Process Dataset", type="primary"):
                    with st.spinner("Processing..."):
                        processed_df = process_dataframe(df, st.session_state.dictionaries, selected_column)
                        st.session_state.processed_df = processed_df
                        st.success("Dataset processed successfully!")
                        st.rerun()
            else:
                st.warning("No text columns found in the dataset.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with col2:
    st.header("📊 Current Dictionaries")
    
    for dict_name, keywords in st.session_state.dictionaries.items():
        with st.expander(f"{dict_name} ({len(keywords)} keywords)"):
            st.write(", ".join(sorted(keywords)))

# --- Results Section --------------------------------------------------------
if st.session_state.processed_df is not None:
    st.header("📈 Classification Results")
    
    df_result = st.session_state.processed_df
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(df_result))
    
    with col2:
        classified_rows = len(df_result[df_result['labels'].apply(len) > 0])
        st.metric("Classified Rows", classified_rows)
    
    with col3:
        classification_rate = (classified_rows / len(df_result)) * 100
        st.metric("Classification Rate", f"{classification_rate:.1f}%")
    
    # Category breakdown
    st.subheader("Category Breakdown")
    category_counts = {}
    for dict_name in st.session_state.dictionaries.keys():
        if dict_name in df_result.columns:
            category_counts[dict_name] = df_result[dict_name].sum()
    
    if category_counts:
        category_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        st.bar_chart(category_df.set_index('Category'))
    
    # Full results
    st.subheader("Full Results")
    st.dataframe(df_result)
    
    # Download processed data
    csv_data = df_result.to_csv(index=False)
    st.download_button(
        label="📥 Download Classified Dataset",
        data=csv_data,
        file_name="classified_dataset.csv",
        mime="text/csv",
        type="primary"
    )

# --- Instructions -----------------------------------------------------------
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Instructions:
    
    1. **Upload Dataset**: Upload a CSV file containing text data you want to classify
    2. **Select Text Column**: Choose which column contains the text to be analyzed
    3. **Manage Dictionaries**: Use the sidebar to:
       - Edit existing keyword dictionaries
       - Add new dictionaries
       - Import/export dictionary configurations
    4. **Process**: Click "Process Dataset" to run the classification
    5. **Review Results**: Examine the classification results and download the processed data
    
    ### Dictionary Format:
    - Each dictionary contains keywords/phrases (case-insensitive)
    - Multiple dictionaries can match the same text
    - Keywords are matched using substring search
    
    ### Output:
    - `labels`: List of matched dictionary names for each row
    - Individual boolean columns for each dictionary (True/False)
    """)

# --- Footer -----------------------------------------------------------------
st.markdown("---")
st.markdown("Built with Streamlit 🎈")
