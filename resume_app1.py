import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import StringIO

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return ""

# Function to extract basic details (name, email, skills) from resume text
def extract_details(text):
    # Extract email
    email = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    email = email[0] if email else "Not Found"

    # Extract skills (example keywords)
    skills_keywords = ["python", "java", "javascript", "html", "css", "sql", "machine learning", "data analysis"]
    skills = [skill for skill in skills_keywords if skill in text.lower()]
    skills = ", ".join(skills) if skills else "Not Found"

    return {"Email": email, "Skills": skills}

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
st.set_page_config(page_title="AI Resume Screening & Candidate Ranking System", layout="wide")
st.title("🚀 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description", height=200)

# File uploader
st.header("📤 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")
    
    resumes = []
    details_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Extract text and details from resumes
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}...")
        text = extract_text_from_pdf(file)
        resumes.append(text)
        details = extract_details(text)
        details["Resume"] = file.name
        details_list.append(details)
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing complete!")

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create a DataFrame with results
    results = pd.DataFrame(details_list)
    results["Score"] = scores
    results = results.sort_values(by="Score", ascending=False)

    # Display results
    st.subheader("📋 Ranked Resumes")
    st.dataframe(results)

    # Highlight keywords in job description
    st.subheader("🔍 Keywords in Job Description")
    keywords = TfidfVectorizer().build_tokenizer()(job_description.lower())
    st.write("Keywords: " + ", ".join(keywords))

    # Download results as CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="ranked_resumes.csv",
        mime="text/csv",
    )

else:
    st.warning("Please upload resumes and enter a job description to proceed.")