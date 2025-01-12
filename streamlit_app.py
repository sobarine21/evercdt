import streamlit as st
import pandas as pd
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from langdetect import detect
from pdfminer.high_level import extract_text
from io import StringIO

# Set up Google API keys
API_KEY = st.secrets["GOOGLE_API_KEY"]
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

# Streamlit UI
st.title("Advanced Copyright Content Detection Tool")
st.sidebar.header("Navigation")
content_type = st.sidebar.radio("Select Content Type", ["Text", "File"])

# Unique Features
st.sidebar.markdown("### Additional Features")
export_csv = st.sidebar.checkbox("Export Results as CSV")
visualize_results = st.sidebar.checkbox("Visualize Results")

# Placeholder for results
results = []

# Language Detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Content Summarization
def summarize_content(text, max_words=50):
    words = text.split()[:max_words]
    return " ".join(words) + "..." if len(words) == max_words else " ".join(words)

# Search and Similarity Detection
def search_and_analyze(user_content):
    service = build("customsearch", "v1", developerKey=API_KEY)
    response = service.cse().list(q=user_content, cx=CX).execute()
    search_results = response.get('items', [])
    detected_matches = []

    for result in search_results:
        url = result['link']
        st.write(f"Analyzing: {url}...")

        try:
            content_response = requests.get(url, timeout=10)
            if content_response.status_code == 200:
                web_content = content_response.text
                soup = BeautifulSoup(web_content, "html.parser")
                paragraphs = soup.find_all("p")
                web_text = " ".join([para.get_text() for para in paragraphs])
                vectorizer = TfidfVectorizer().fit_transform([user_content, web_text])
                similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
                if similarity > 0.5:  # Threshold for matches
                    detected_matches.append({
                        "URL": url,
                        "Similarity": round(similarity * 100, 2),
                        "Summary": summarize_content(web_text)
                    })
        except Exception as e:
            st.error(f"Error processing URL {url}: {e}")

    return detected_matches

# Handle Text Input
if content_type == "Text":
    user_content = st.text_area("Paste your copyrighted content:", height=200)
    if user_content:
        lang = detect_language(user_content)
        st.write(f"Detected Language: {lang}")
        if st.button("Search for Violations"):
            results = search_and_analyze(user_content)

# Handle File Upload
elif content_type == "File":
    uploaded_file = st.file_uploader("Upload a text file or PDF:", type=["txt", "pdf"])
    if uploaded_file:
        file_content = ""
        if uploaded_file.type == "text/plain":
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text(uploaded_file)
        st.write("File Content:")
        st.write(file_content)

        if st.button("Search for Violations"):
            results = search_and_analyze(file_content)

# Display Results
if results:
    st.success(f"Found {len(results)} potential matches.")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    if visualize_results:
        st.bar_chart(results_df.set_index("URL")["Similarity"])

    if export_csv:
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Results as CSV", data=csv, file_name="results.csv", mime="text/csv")
