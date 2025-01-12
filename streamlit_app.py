import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from io import StringIO
from langdetect import detect
from pdfminer.high_level import extract_text

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

# Streamlit UI for text input and file upload
st.title("Advanced Copyright Content Detection Tool")
st.write("Detect if your copyrighted content is being used elsewhere on the web.")

# Option for user to choose content type: text or file
content_type = st.selectbox("Select Content Type", ["Text", "File"])

# Function to search the web for potential copyright violations
def search_web(content):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        response = service.cse().list(q=content, cx=CX).execute()
        
        # Extract URLs from the search results
        search_results = response.get('items', [])
        detected_matches = []

        for result in search_results:
            url = result['link']
            st.write(f"Analyzing {url}...")

            # Fetch the content from the URL
            content_response = requests.get(url, timeout=10)
            if content_response.status_code == 200:
                web_content = content_response.text

                # Clean and parse the HTML content
                soup = BeautifulSoup(web_content, "html.parser")
                paragraphs = soup.find_all("p")
                web_text = " ".join([para.get_text() for para in paragraphs])

                # Calculate similarity between user content and web content
                vectorizer = TfidfVectorizer().fit_transform([content, web_text])
                similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                # If similarity exceeds a threshold, record the match
                if similarity[0][0] > 0.8:  # Adjust the threshold as needed
                    detected_matches.append((url, similarity[0][0]))

        return detected_matches
    except Exception as e:
        st.error(f"Error: {e}")
        return []

# Handle Text Input
if content_type == "Text":
    user_content = st.text_area("Paste your copyrighted content:", height=200)

    # Language detection for multilingual content
    if user_content:
        lang = detect(user_content)
        st.write(f"Detected language: {lang}")

    if st.button("Search the Web for Copyright Violations"):
        if not user_content.strip():
            st.error("Please provide your copyrighted content.")
        else:
            matches = search_web(user_content)
            if matches:
                st.success("Potential copyright violations detected!")
                for match in matches:
                    st.write(f"- **URL**: {match[0]} - **Similarity**: {match[1]:.2f}")
            else:
                st.info("No matches found.")

# Handle File Upload (PDF, TXT)
elif content_type == "File":
    uploaded_file = st.file_uploader("Upload a text document to analyze:", type=["txt", "pdf"])

    if uploaded_file is not None:
        file_content = ""
        if uploaded_file.type == "text/plain":
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text(uploaded_file)

        st.write("Content from uploaded file:")
        st.write(file_content)

        # Option to perform search after uploading
        if st.button("Search the Web for Copyright Violations"):
            if not file_content.strip():
                st.error("The uploaded file has no content.")
            else:
                matches = search_web(file_content)
                if matches:
                    st.success("Potential copyright violations detected!")
                    for match in matches:
                        st.write(f"- **URL**: {match[0]} - **Similarity**: {match[1]:.2f}")
                else:
                    st.info("No matches found.")
