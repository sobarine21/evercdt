import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from io import StringIO
from langdetect import detect
from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

# Dashboard to display results
def display_dashboard(results):
    st.subheader("Search Results Dashboard")
    if results:
        for i, (url, similarity) in enumerate(results):
            st.write(f"**Result {i+1}:**")
            st.write(f"- **URL**: {url}")
            st.write(f"- **Similarity Score**: {similarity:.2f}")
            st.write("---")
    else:
        st.info("No matches found.")

# Perform web search and analyze content
def perform_web_search(content, threshold=0.8):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        response = service.cse().list(q=content, cx=CX).execute()
        search_results = response.get('items', [])
        
        detected_matches = []
        for result in search_results:
            url = result['link']
            st.write(f"Analyzing {url}...")
            content_response = requests.get(url, timeout=10)
            if content_response.status_code == 200:
                soup = BeautifulSoup(content_response.text, "html.parser")
                paragraphs = soup.find_all("p")
                web_text = " ".join([para.get_text() for para in paragraphs])

                vectorizer = TfidfVectorizer().fit_transform([content, web_text])
                similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                if similarity[0][0] > threshold:
                    detected_matches.append((url, similarity[0][0]))
        return detected_matches
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Streamlit UI
def main():
    st.title("Advanced Copyright Content Detection Tool")
    st.write("Detect if your copyrighted content is being used elsewhere on the web.")

    content_type = st.selectbox("Select Content Type", ["Text", "Image", "File"])

    if content_type == "Text":
        user_content = st.text_area("Paste your copyrighted content:", height=200)
        if user_content:
            lang = detect(user_content)
            st.write(f"Detected language: {lang}")
        if st.button("Search the Web for Copyright Violations"):
            if not user_content.strip():
                st.error("Please provide your copyrighted content.")
            else:
                results = perform_web_search(user_content)
                display_dashboard(results)

    elif content_type == "Image":
        uploaded_image = st.file_uploader("Upload an image to analyze:", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                st.write("Extracted Text:")
                st.write(extracted_text)
                if st.button("Search the Web for Copyright Violations"):
                    results = perform_web_search(extracted_text)
                    display_dashboard(results)
            else:
                st.error("No text detected in the image.")

    elif content_type == "File":
        uploaded_file = st.file_uploader("Upload a text document to analyze:", type=["txt", "pdf", "docx"])
        if uploaded_file:
            file_content = ""
            if uploaded_file.type == "text/plain":
                file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            elif uploaded_file.type == "application/pdf":
                file_content = extract_text(uploaded_file)
            # Placeholder for DOCX support

            if file_content.strip():
                st.write("Content from uploaded file:")
                st.write(file_content)
                if st.button("Search the Web for Copyright Violations"):
                    results = perform_web_search(file_content)
                    display_dashboard(results)
            else:
                st.error("The uploaded file has no content.")

if __name__ == "__main__":
    main()
