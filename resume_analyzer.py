import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import pdfminer.high_level
from pdfminer.layout import LAParams
import os
import spacy
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK resources (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfminer.six."""
    try:
        output_string = StringIO()
        with open(pdf_path, 'rb') as in_file:
            pdfminer.high_level.extract_text_to_fp(in_file, output_string, laparams=LAParams())
        return output_string.getvalue()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


def preprocess_text(text):
    """Preprocesses text by removing punctuation, converting to lowercase,
    tokenizing, and removing stopwords.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Get stopwords
    filtered_tokens = [w for w in tokens if not w in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)  # Join back into a string


def extract_name(resume_text):
    """Attempts to extract the name from the resume text using spaCy."""
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Name not found"


def extract_phone_number(resume_text):
    """Extract phone number using a regex pattern."""
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, resume_text)
    return match.group() if match else None


def extract_email(resume_text):
    """Extract email address using a regex pattern."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, resume_text)
    return match.group() if match else None


def extract_experience_years(resume_text):
    """Extract total years of experience from resume text."""
    experience_years = re.findall(r'(\d+)\+?\s+(?:years|yrs)', resume_text, re.I)
    if experience_years:
        return max(map(int, experience_years))
    return None


def analyze_resume_cpi(resume_path, job_description, resume_text):
    """Analyzes a resume against a job description, specifically for SAP CPI expertise."""
    processed_resume = preprocess_text(resume_text)
    processed_job_description = preprocess_text(job_description)

    # Calculate similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_resume, processed_job_description])
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Keyword Matching (Basic)
    job_keywords = processed_job_description.split()
    resume_keywords = processed_resume.split()
    matched_keywords = list(set(job_keywords) & set(resume_keywords))  # Find common keywords

    # SAP CPI Specific Skills Analysis
    cpi_skills = [
        "sap cpi", "sap cloud platform integration", "cloud platform integration",
        "integration flow", "iflow", "odata", "soap", "rest", "sftp", "http",
        "successfactors", "s4hana", "ariba", "fieldglass", "sap erp",
        "groovy script", "xslt mapping", "message mapping", "value mapping",
        "content modifier", "router", "process direct", "request reply",
        "integration adapter", "api management", "security artifacts",
        "certificate", "keystore", "oauth", "cpi monitoring", "hci", "hana cloud integration",
        "cloud foundry", "scp", "sap cloud platform", "api proxy",
        "camel", "apache camel", "edi", "idoc", "as2", "xpath", "json",
        "bpm", "business process management", "cpi administration",
        "transport management", "cpi security", "odata services", "sap analytics cloud",
        "bapi", "rfc", "s4/hana", "cpi developer", "successfactors integration", "api management",
        "camel context", "message queue", "cloud integration", "integration suite"
    ]

    cpi_skills_found = [skill for skill in cpi_skills if skill in processed_resume]

    # Seniority Keywords (Adjust as needed)
    seniority_keywords = ["lead", "architect", "senior", "expert", "consultant", "principal"]
    seniority_found = any(keyword in processed_resume for keyword in seniority_keywords)

    # Certification Keywords
    cert_keywords = ["certified", "certificate"]
    certs_found = any(keyword in processed_resume for keyword in cert_keywords)

    analysis_results = {
        "similarity_score": similarity_score,
        "matched_keywords": matched_keywords,
        "cpi_skills_found": cpi_skills_found,
        "seniority_level": "Senior" if seniority_found else "Intermediate/Junior",
        "overall_fit": "Good" if similarity_score > 0.4 and seniority_found and certs_found else "Average",
        "certifications": "Present" if certs_found else "Absent"
    }

    return analysis_results


def get_files_from_user():
    """Prompt user for file or folder path."""
    user_input = input("Enter the file path or folder path: ")
    if os.path.isdir(user_input):
        files = [os.path.join(user_input, f) for f in os.listdir(user_input) if f.lower().endswith(('.pdf', '.txt'))]
        if not files:
            print("No PDF or TXT files found in the folder.")
        return files
    elif os.path.isfile(user_input):
        return [user_input]
    else:
        print("Invalid file or folder path.")
        return []


def save_to_excel(results, output_file):
    """Save results to an Excel file."""
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)


def analyze_resume_for_job(resume_path, job_description):
    """Main function to analyze a single resume."""
    try:
        # Read resume text
        if resume_path.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_path)
        else:
            with open(resume_path, "r", encoding="utf-8") as f:
                resume_text = f.read()

        if not resume_text:
            logging.error(f"Error: Could not extract text from {resume_path}.")
            return

        # Extract details
        name = extract_name(resume_text)
        phone = extract_phone_number(resume_text)
        email = extract_email(resume_text)
        experience_years = extract_experience_years(resume_text)

        # Analyze resume
        analysis_results = analyze_resume_cpi(resume_path, job_description, resume_text)

        # Collect results
        result = {
            "Name": name,
            "Phone": phone,
            "Email": email,
            "Total Experience (Years)": experience_years,
            "Similarity Score": analysis_results['similarity_score'],
            "Matched Keywords": ', '.join(analysis_results['matched_keywords']),
            "CPI Skills Found": ', '.join(analysis_results['cpi_skills_found']),
            "Seniority Level": analysis_results['seniority_level'],
            "Overall Fit": analysis_results['overall_fit'],
            "Certifications": analysis_results['certifications']
        }
        return result

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':
    job_description = """We are seeking a Senior SAP CPI Consultant to design and implement integration solutions. Integration iflows, JMS message queues, event broker, event mesh, api management
                         """

    # Ask user for file or folder path
    files_to_process = get_files_from_user()

    results = []
    if files_to_process:
        for resume_path in files_to_process:
            logging.info(f"Processing: {resume_path}")
            result = analyze_resume_for_job(resume_path, job_description)
            if result:
                results.append(result)

        if results:
            # Save results to Excel
            output_file = "resume_analysis_results.xlsx"
            save_to_excel(results, output_file)
            logging.info(f"Results saved to {output_file}")
