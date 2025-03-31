import streamlit as st
import numpy as np
import json
import assemblyai as aai
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import time
import plotly.express as px
import pandas as pd
from datetime import datetime
import tempfile

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Audio to Subtitles Search",
    layout="wide",
    page_icon="🎬"
)

# Initialize session state variables
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = "{}"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

# Securely handle AssemblyAI API key
def get_api_key():
    if "general" in st.secrets and "ASSEMBLYAI_API_KEY" in st.secrets["general"]:
        return st.secrets["general"]["ASSEMBLYAI_API_KEY"]
    else:
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if api_key:
            return api_key
        else:
            st.error("⚠️ API key not found! Please add it to Streamlit secrets or environment variables.")
            st.stop()

# Set the API key
aai.settings.api_key = get_api_key()

# Simple Vector Database
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def add(self, documents, metadatas=None, embeddings=None):
        if embeddings is None:
            embeddings = self.model.encode(documents, show_progress_bar=False)
        
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.embeddings.append(embeddings[i])
            self.metadata.append(metadatas[i] if metadatas else {})
            
    def query(self, query_embeddings, n_results=5):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        query_embedding = query_embeddings[0]
        similarities = []
        for emb in self.embeddings:
            dot_product = sum(a*b for a, b in zip(query_embedding, emb))
            magnitude1 = sum(a*a for a in query_embedding) ** 0.5
            magnitude2 = sum(b*b for b in emb) ** 0.5
            sim = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
            similarities.append(sim)
            
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:n_results]
        
        result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for idx in top_indices:
            result["documents"][0].append(self.documents[idx])
            result["metadatas"][0].append(self.metadata[idx])
            result["distances"][0].append(similarities[idx])
            
        return result

# Initialize vector database
@st.cache_resource
def get_vector_db():
    return SimpleVectorDB()

db = get_vector_db()

# Sample data
sample_subtitles = [
    "I am Groot. I am Groot. I am Groot.",
    "May the Force be with you.",
    "Houston, we have a problem.",
    "My precious... my precious...",
    "Life is like a box of chocolates.",
    "I'll be back.",
    "Bond. James Bond.",
    "There's no place like home.",
    "E.T. phone home.",
    "To infinity and beyond!"
]

sample_metadata = [
    {"subtitle_name": "Guardians of the Galaxy", "subtitle_id": "12345", "year": "2014", "genre": "Sci-Fi"},
    {"subtitle_name": "Star Wars", "subtitle_id": "23456", "year": "1977", "genre": "Sci-Fi"},
    {"subtitle_name": "Apollo 13", "subtitle_id": "34567", "year": "1995", "genre": "Drama"},
    {"subtitle_name": "The Lord of the Rings", "subtitle_id": "45678", "year": "2001", "genre": "Fantasy"},
    {"subtitle_name": "Forrest Gump", "subtitle_id": "56789", "year": "1994", "genre": "Drama"},
    {"subtitle_name": "The Terminator", "subtitle_id": "67890", "year": "1984", "genre": "Action"},
    {"subtitle_name": "James Bond Series", "subtitle_id": "78901", "year": "1962", "genre": "Action"},
    {"subtitle_name": "The Wizard of Oz", "subtitle_id": "89012", "year": "1939", "genre": "Fantasy"},
    {"subtitle_name": "E.T.", "subtitle_id": "90123", "year": "1982", "genre": "Sci-Fi"},
    {"subtitle_name": "Toy Story", "subtitle_id": "01234", "year": "1995", "genre": "Animation"}
]

if not db.documents:
    db.add(sample_subtitles, sample_metadata)

def transcribe_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file.", None, None
    
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getbuffer())
            tmp_path = tmp_file.name
        
        config = aai.TranscriptionConfig(
            language_code="en",
            punctuate=True,
            format_text=True
        )
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(tmp_path)
        
        os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        confidence = None
        if hasattr(transcript, 'words') and transcript.words:
            confidences = [word.confidence for word in transcript.words if hasattr(word, 'confidence')]
            if confidences:
                confidence = sum(confidences) / len(confidences)
                
        return transcript.text, transcript.text, confidence, processing_time
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return f"Transcription error: {str(e)}", None, None, time.time() - start_time

def retrieve_and_display_results(query, top_n):
    if not query:
        return json.dumps([{"Result": "No transcription text available for search."}], indent=4)

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([query], show_progress_bar=False)
        results = db.query(query_embeddings=query_embedding, n_results=top_n)
        return format_results_as_json(results)
    except Exception as e:
        return json.dumps([{"Result": f"Search error: {e}"}], indent=4)

def format_results_as_json(results):
    formatted_results = []
    if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
        for i in range(len(results["documents"][0])):
            subtitle_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            similarity_score = results["distances"][0][i] if "distances" in results else None
            
            formatted_results.append({
                "Result": i + 1,
                "Similarity": f"{similarity_score:.4f}" if similarity_score is not None else "N/A",
                "Media Title": metadata.get("subtitle_name", "Unknown").upper(),
                "Year": metadata.get("year", "Unknown"),
                "Genre": metadata.get("genre", "Unknown"),
                "Quote": subtitle_text,
                "URL": f"https://www.opensubtitles.org/en/subtitles/{metadata.get('subtitle_id', 'N/A')}"
            })
        return json.dumps(formatted_results, indent=4)
    return json.dumps([{"Result": "No matching subtitles found"}], indent=4)

def clear_all():
    st.session_state.transcribed_text = ""
    st.session_state.search_results = "{}"
    st.session_state.confidence_score = None
    st.session_state.processing_time = None

def save_search_history(query, results_json):
    results = json.loads(results_json)
    first_result = results[0] if results else {"Media Title": "No results"}
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query[:50] + "..." if len(query) > 50 else query,
        "top_result": first_result.get("Media Title", "Unknown") if "Result" not in first_result else "No matches",
        "num_results": len(results) if "Result" not in results[0] else 0
    }
    st.session_state.search_history.append(history_entry)
    if len(st.session_state.search_history) > 20:
        st.session_state.search_history = st.session_state.search_history[-20:]

def display_metrics():
    col1, col2 = st.columns(2)
    if st.session_state.processing_time:
        with col1:
            st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
    if st.session_state.confidence_score:
        with col2:
            st.metric("Transcription Confidence", f"{st.session_state.confidence_score*100:.1f}%")

def display_search_history():
    if not st.session_state.search_history:
        st.info("No search history available yet.")
        return
        
    st.subheader("📊 Search History")
    history_df = pd.DataFrame(st.session_state.search_history)
    st.dataframe(history_df, use_container_width=True)
    
    if len(history_df) >= 3:
        result_counts = history_df['top_result'].value_counts().reset_index()
        result_counts.columns = ['Media', 'Count']
        fig = px.bar(result_counts, x='Media', y='Count', title='Most Frequent Matches',
                    labels={'Media': 'Media Title', 'Count': 'Number of Matches'},
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

def display_theme_customization():
    st.sidebar.subheader("🎨 Theme Customization")
    themes = {
        "Default": {"primary": "#F63366", "background": "#FFFFFF", "text": "#262730"},
        "Dark Mode": {"primary": "#00CCFF", "background": "#262730", "text": "#FAFAFA"}
    }
    selected_theme = st.sidebar.selectbox("Select Theme", list(themes.keys()))
    theme = themes[selected_theme]
    css = f"""
    <style>
        .stApp {{
            background-color: {theme['background']};
            color: {theme['text']};
        }}
        .stButton>button {{
            background-color: {theme['primary']};
            color: white;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.title("🔍 Audio to Subtitles Search")
    st.markdown("### Search for matching subtitles from audio clips")
    
    display_theme_customization()
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 History", "ℹ️ About"])
    
    with tab1:
        with st.sidebar:
            st.header("⚙️ Settings")
            top_n_results = st.slider("Results to Display:", min_value=1, max_value=10, value=5)
            
            st.subheader("📂 Upload Audio")
            audio_input = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a", "ogg"])
            
            if audio_input:
                st.audio(audio_input, format=f"audio/{audio_input.name.split('.')[-1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Transcribe and Search"):
                    with st.spinner("Processing audio..."):
                        if audio_input:
                            transcribed_text, raw_text, confidence, proc_time = transcribe_audio(audio_input)
                            st.session_state.transcribed_text = transcribed_text
                            st.session_state.confidence_score = confidence
                            st.session_state.processing_time = proc_time
                            
                            if transcribed_text and not transcribed_text.startswith("Error"):
                                result_json = retrieve_and_display_results(transcribed_text, top_n_results)
                                st.session_state.search_results = result_json
                                save_search_history(transcribed_text, result_json)
                            else:
                                st.session_state.search_results = json.dumps([{"Result": "Transcription failed"}], indent=4)
            with col2:
                if st.button("🧹 Clear"):
                    clear_all()
                    st.rerun()
        
        display_metrics()
        
        st.subheader("📝 Transcribed Text")
        st.text_area("", value=st.session_state.transcribed_text, height=100)
        
        st.subheader("🔍 Matching Subtitles")
        try:
            results = json.loads(st.session_state.search_results)
            if results and isinstance(results, list) and len(results) > 0:
                if "Result" in results[0] and results[0]["Result"] == "No matching subtitles found":
                    st.info("No matching subtitles found.")
                elif "Result" not in results[0] or isinstance(results[0]["Result"], int):
                    for i, result in enumerate(results):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i + 1}. 🎥 {result.get('Media Title', 'Unknown')}**")
                            st.markdown(f"📜 *\"{result.get('Quote', '')}\"*")
                            st.markdown(f"📅 Year: {result.get('Year', 'Unknown')} | 🎭 Genre: {result.get('Genre', 'Unknown')}")
                            st.markdown(f"📊 Similarity: {result.get('Similarity', 'N/A')}")
                        with col2:
                            st.markdown(f"🔗 [View Subtitle]({result.get('URL', '#')})")
                        st.markdown("---")
                else:
                    st.warning(results[0].get("Result", "Unknown error"))
            else:
                st.info("Upload and analyze audio to see results.")
        except Exception as e:
            st.warning(f"Error displaying results: {str(e)}")
    
    with tab2:
        display_search_history()
    
    with tab3:
        st.subheader("About This Application")
        st.markdown("""
        ### Audio to Subtitles Search
        
        This application transcribes audio files and searches for matching subtitles using semantic search.
        
        #### How It Works:
        1. Upload an audio file
        2. Audio is transcribed using AssemblyAI
        3. Transcript is matched against a subtitle database
        4. Top matching results are displayed with links
        
        #### Features:
        - Real-time audio transcription
        - Semantic search for similar quotes
        - Performance metrics
        - Search history tracking
        - Theme customization
        
        #### Limitations:
        - Demo uses sample database
        - Works best with clear audio
        - English language only
        """)

if __name__ == "__main__":
    main()
