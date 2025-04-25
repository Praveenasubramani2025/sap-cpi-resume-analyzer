import os
import subprocess

# Ensure spaCy model is downloaded
try:
    import en_core_web_sm
except ImportError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
import streamlit as st
import pandas as pd
import os
import tempfile
from resume_analyzer import analyze_resume_for_job, save_to_excel

st.set_page_config(page_title="SAP CPI Resume Analyzer", layout="wide")

st.title("📄 SAP CPI Resume Analyzer")
st.write("Upload multiple resumes and get a downloadable analysis report based on the job description.")

job_description = st.text_area("Enter the Job Description", height=200)

uploaded_files = st.file_uploader("Upload resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Analyze Resumes") and uploaded_files and job_description:
    with st.spinner("Analyzing..."):
        results = []
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            result = analyze_resume_for_job(file_path, job_description)
            if result:
                results.append(result)

        if results:
            df = pd.DataFrame(results)
            st.success(f"✅ {len(results)} resume(s) analyzed.")
            st.dataframe(df)

            output_file = os.path.join(temp_dir, "resume_analysis_results.xlsx")
            save_to_excel(results, output_file)
            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=f,
                    file_name="resume_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No results to display.")
