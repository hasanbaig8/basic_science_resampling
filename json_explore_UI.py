import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Page config
st.set_page_config(
    page_title="JSON Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vibe-coded aesthetics
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D3748 0%, #1A202C 100%);
    }
    h1, h2, h3 {
        color: #E2E8F0;
        font-weight: 700;
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    .json-entry {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def load_json_files(directory: str) -> Dict[str, List[Dict]]:
    """Load all JSON files from the directory"""
    json_files = {}
    for file_path in Path(directory).glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                json_files[file_path.name] = data
        except Exception as e:
            st.error(f"Error loading {file_path.name}: {e}")
    return json_files

def get_statistics(data: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from the data"""
    stats = {
        'total_entries': len(data),
        'classifier_true': sum(1 for entry in data if entry.get('classifier_completion_bool', False)),
        'classifier_false': sum(1 for entry in data if not entry.get('classifier_completion_bool', False)),
        'models': set(entry.get('model_name', 'Unknown') for entry in data)
    }
    return stats

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>🔍 JSON Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #E2E8F0; font-size: 18px;'>Explore blackmail experiment data with style</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📁 File Selection")
    data_dir = "/workspace/basic_science_resampling/blackmail_data"
    
    # Load all JSON files
    json_files = load_json_files(data_dir)
    
    if not json_files:
        st.error("No JSON files found in the directory!")
        return
    
    # File selector
    selected_file = st.sidebar.selectbox(
        "Choose a JSON file:",
        options=list(json_files.keys()),
        format_func=lambda x: x.replace('.json', '').replace('_', ' ').title()
    )
    
    if selected_file:
        data = json_files[selected_file]
        stats = get_statistics(data)
        
        # Statistics section
        st.markdown("### 📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{stats['total_entries']}</h2>
                <p>Total Entries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{stats['classifier_true']}</h2>
                <p>Classified True</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{stats['classifier_false']}</h2>
                <p>Classified False</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            percentage = (stats['classifier_true'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h2>{percentage:.1f}%</h2>
                <p>True Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sidebar filters
        st.sidebar.markdown("### 🎛️ Filters")
        
        # Filter by classifier result
        filter_option = st.sidebar.radio(
            "Filter by classification:",
            ["All", "True only", "False only"]
        )
        
        # Apply filters
        filtered_data = data
        if filter_option == "True only":
            filtered_data = [entry for entry in data if entry.get('classifier_completion_bool', False)]
        elif filter_option == "False only":
            filtered_data = [entry for entry in data if not entry.get('classifier_completion_bool', False)]
        
        st.sidebar.markdown(f"**Showing:** {len(filtered_data)} entries")
        
        # Search functionality
        st.sidebar.markdown("### 🔎 Search")
        search_term = st.sidebar.text_input("Search in responses:", "")
        
        if search_term:
            filtered_data = [
                entry for entry in filtered_data 
                if search_term.lower() in str(entry.get('full_response', '')).lower()
            ]
            st.sidebar.markdown(f"**Found:** {len(filtered_data)} matches")
        
        # Entry selector
        st.sidebar.markdown("### 📄 Select Entry")
        entry_index = st.sidebar.number_input(
            "Entry number:",
            min_value=0,
            max_value=max(0, len(filtered_data) - 1),
            value=0,
            step=1
        )
        
        # Display the selected entry
        if filtered_data:
            entry = filtered_data[entry_index]
            
            st.markdown(f"### Entry {entry_index + 1} of {len(filtered_data)}")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Response", "📨 Prompt Data", "⚙️ Metadata", "🔍 Raw JSON"])
            
            with tab1:
                st.markdown("#### AI Response")
                response = entry.get('full_response', 'No response available')
                
                # Check if classifier is true or false
                is_true = entry.get('classifier_completion_bool', False)
                status_color = "🟢" if is_true else "🔴"
                st.markdown(f"**Classification:** {status_color} {is_true}")
                
                st.text_area("Response:", response, height=400)
            
            with tab2:
                prompt_data = entry.get('prompt_data', {})
                
                st.markdown("#### System Prompt")
                st.text_area("System:", prompt_data.get('system_prompt', ''), height=200)
                
                st.markdown("#### User Prompt")
                st.text_area("User:", prompt_data.get('user_prompt', ''), height=150)
                
                st.markdown("#### Email Content")
                with st.expander("Show full email content"):
                    st.text_area("Emails:", prompt_data.get('email_content', ''), height=400)
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Name:**")
                    st.code(entry.get('model_name', 'Unknown'))
                    
                    st.markdown("**Classifier Model:**")
                    st.code(entry.get('classifier_model_name', 'Unknown'))
                
                with col2:
                    st.markdown("**Classifier Result:**")
                    result = entry.get('classifier_completion_bool', False)
                    st.code(str(result))
            
            with tab4:
                st.json(entry)
        else:
            st.warning("No entries match your filters!")
        
        # Download section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 Export")
        if st.sidebar.button("Download Filtered Data"):
            json_str = json.dumps(filtered_data, indent=2)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"filtered_{selected_file}",
                mime="application/json"
            )

if __name__ == "__main__":
    main()

