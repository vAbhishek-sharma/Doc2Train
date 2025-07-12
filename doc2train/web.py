### This is just a Proof of concept right Now  ###
### Further research is needed                 ###
# doc2train/web.py - Streamlit Web UI
"""
Beautiful Streamlit Web UI for Doc2Train
"""

import streamlit as st
import requests
import time
import json
from pathlib import Path
import tempfile
import zipfile
import io

# Configure Streamlit
st.set_page_config(
    page_title="Doc2Train",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🚀 Doc2Train</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Convert your documents to AI training data - no coding required!</p>', unsafe_allow_html=True)

    # Check if API is running
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000", help="URL of the Doc2Train API server")

    if not check_api_connection(api_url):
        st.error(f"""
        🚨 **API Server Not Running**

        Please start the API server first:
        ```bash
        doc2train-api
        ```

        Or install and start:
        ```bash
        pip install doc2train
        doc2train-api
        ```
        """)
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key
        api_key = st.text_input(
            "🔑 API Key",
            type="password",
            help="OpenAI, Anthropic, or Google API key"
        )

        # Processing mode
        mode = st.selectbox(
            "🎯 What do you want to do?",
            [
                ("extract_only", "📄 Extract text only (free)"),
                ("generate", "🤖 Generate training data (requires API key)"),
                ("direct_media", "🎬 Analyze images/videos (requires API key)")
            ],
            format_func=lambda x: x[1]
        )[0]

        # Provider selection
        providers = get_available_providers(api_url)
        provider = st.selectbox("🔧 AI Provider", providers) if providers else "openai"

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            generators = st.multiselect(
                "Training Data Types",
                ["conversations", "qa_pairs", "summaries", "embeddings"],
                default=["conversations", "qa_pairs"]
            )

            chunk_size = st.slider("Chunk Size", 1000, 8000, 4000)
            quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.7)
            output_format = st.selectbox("Output Format", ["jsonl", "csv", "txt"])

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to process",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'pptx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported: PDF, Word, PowerPoint, text files, and images"
        )

        # Processing button
        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_documents(
                api_url, uploaded_files, mode, provider, api_key,
                generators, chunk_size, quality_threshold, output_format
            )

    with col2:
        st.header("🎯 Quick Start")

        # Quick start guide
        st.markdown("""
        **🆕 First time using Doc2Train?**

        1. 📁 Upload a document (drag & drop works!)
        2. 🔑 Add your API key in the sidebar
        3. 🎯 Choose what you want to generate
        4. 🚀 Click "Process Documents"

        **💡 No API key?**

        - Try "Extract text only" first (it's free!)
        - Get OpenAI key: [platform.openai.com](https://platform.openai.com/api-keys)
        - Get Anthropic key: [console.anthropic.com](https://console.anthropic.com/)
        """)

        # Sample files
        st.header("📚 Try Sample Files")

        sample_files = {
            "📄 Sample Article": create_sample_text(),
            "📊 Sample Report": create_sample_report(),
            "📝 Sample Notes": create_sample_notes()
        }

        for name, content in sample_files.items():
            if st.button(name, use_container_width=True):
                st.download_button(
                    f"Download {name}",
                    content,
                    f"{name.lower().replace(' ', '_')}.txt",
                    "text/plain",
                    use_container_width=True
                )

        # Usage statistics
        if st.button("📊 Show Usage Stats", use_container_width=True):
            show_usage_stats(api_url)

def check_api_connection(api_url: str) -> bool:
    """Check if API server is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_providers(api_url: str) -> list:
    """Get available LLM providers from API"""
    try:
        response = requests.get(f"{api_url}/providers", timeout=5)
        if response.status_code == 200:
            return response.json().get("providers", ["openai"])
    except:
        pass
    return ["openai", "anthropic", "google"]

def process_documents(api_url, uploaded_files, mode, provider, api_key, generators, chunk_size, quality_threshold, output_format):
    """Process uploaded documents via API"""

    # Validation
    if mode != "extract_only" and not api_key:
        st.error("🔑 API key required for AI generation. Please add your key in the sidebar.")
        return

    # Prepare files for upload
    files = []
    for uploaded_file in uploaded_files:
        files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))

    # Prepare configuration
    config_data = {
        "mode": mode,
        "provider": provider,
        "generators": generators,
        "chunk_size": chunk_size,
        "quality_threshold": quality_threshold,
        "output_format": output_format,
        "api_key": api_key
    }

    # Submit job
    with st.spinner("Submitting processing job..."):
        try:
            response = requests.post(
                f"{api_url}/process",
                files=files,
                data={"config": json.dumps(config_data)},
                timeout=30
            )

            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data["job_id"]

                st.success(f"✅ Job submitted successfully! Job ID: {job_id}")

                # Monitor job progress
                monitor_job_progress(api_url, job_id)

            else:
                st.error(f"❌ Error submitting job: {response.text}")

        except Exception as e:
            st.error(f"❌ Connection error: {e}")

def monitor_job_progress(api_url: str, job_id: str):
    """Monitor job progress and show results"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        try:
            response = requests.get(f"{api_url}/status/{job_id}")
            if response.status_code == 200:
                job_status = response.json()

                # Update progress
                progress = job_status["progress"]
                status = job_status["status"]
                message = job_status["message"]

                progress_bar.progress(progress)
                status_text.text(f"Status: {status} - {message}")

                if status == "completed":
                    st.success("🎉 Processing completed successfully!")

                    # Show results
                    result = job_status.get("result")
                    if result:
                        show_results(api_url, job_id, result)
                    break

                elif status == "failed":
                    error = job_status.get("error", "Unknown error")
                    st.error(f"❌ Processing failed: {error}")
                    break

                # Wait before next check
                time.sleep(2)

            else:
                st.error("❌ Error checking job status")
                break

        except Exception as e:
            st.error(f"❌ Error monitoring job: {e}")
            break

def show_results(api_url: str, job_id: str, result: dict):
    """Display processing results"""

    st.header("📊 Processing Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Files Processed", result.get("files_processed", 0))
    with col2:
        st.metric("Success Rate", f"{result.get('success_rate', 0):.1%}")
    with col3:
        st.metric("Total Text", f"{result.get('total_text_chars', 0):,} chars")
    with col4:
        st.metric("Processing Time", f"{result.get('total_processing_time', 0):.1f}s")

    # Download results
    if st.button("📥 Download Results", type="primary"):
        try:
            response = requests.get(f"{api_url}/download/{job_id}")
            if response.status_code == 200:
                st.download_button(
                    "💾 Save Results File",
                    response.content,
                    f"doc2train_results_{job_id}.jsonl",
                    "application/json"
                )
            else:
                st.error("❌ Error downloading results")
        except Exception as e:
            st.error(f"❌ Download error: {e}")

    # Show sample results
    if result.get("sample_outputs"):
        st.subheader("📋 Sample Generated Data")

        for i, sample in enumerate(result["sample_outputs"][:3]):
            with st.expander(f"Sample {i+1}"):
                st.json(sample)

def create_sample_text() -> str:
    """Create sample text document"""
    return """Understanding Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

Key Concepts:
- Supervised Learning: Learning from labeled training data
- Unsupervised Learning: Finding hidden patterns in data
- Reinforcement Learning: Learning through trial and error

Applications:
Machine learning powers recommendation systems, image recognition, natural language processing, and autonomous vehicles.

The field continues to evolve rapidly with advances in deep learning and neural networks.
"""

def create_sample_report() -> str:
    """Create sample report document"""
    return """Q4 2024 Sales Analysis Report

Executive Summary:
Our Q4 performance exceeded expectations with 15% growth over Q3.

Key Metrics:
- Revenue: $2.4M (up 15%)
- New Customers: 342 (up 22%)
- Customer Retention: 94% (up 3%)

Regional Performance:
- North America: $1.2M
- Europe: $800K
- Asia Pacific: $400K

Recommendations:
1. Expand European operations
2. Invest in customer success programs
3. Launch new product line in Q1 2025

This data demonstrates strong market position and growth potential.
"""

def create_sample_notes() -> str:
    """Create sample notes document"""
    return """Meeting Notes - Product Planning Session

Date: March 15, 2024
Attendees: Sarah (PM), Mike (Engineering), Lisa (Design)

Discussion Points:
- User feedback indicates need for mobile app
- Current web platform has 89% satisfaction rate
- Competition launching similar features Q2

Action Items:
1. Sarah: Research mobile development costs
2. Mike: Assess technical feasibility
3. Lisa: Create mobile wireframes

Next Meeting: March 22, 2024

Key Decisions:
- Prioritize mobile development
- Allocate 2 engineers to project
- Target Q3 launch date
"""

def show_usage_stats(api_url: str):
    """Show usage statistics"""
    st.subheader("📊 Usage Statistics")

    # Mock statistics (in real app, get from API)
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents Processed Today", "142")
        st.metric("Total API Calls", "2,847")

    with col2:
        st.metric("Active Users", "28")
        st.metric("Success Rate", "94.2%")

def start_ui(port: int = 8501):
    """Start the Streamlit UI"""
    print(f"""
🌐 Starting Doc2Train Web UI

🎨 Web Interface: http://localhost:{port}
📡 Make sure API server is running: doc2train-api

Ready for drag-and-drop document processing!
""")

    # Note: In practice, you'd use subprocess to start Streamlit
    # subprocess.run(["streamlit", "run", __file__, "--server.port", str(port)])

if __name__ == "__main__":
    main()

