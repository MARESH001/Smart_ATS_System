import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import zipfile
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# === Load Environment Variables ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_KEY"))

# === File to store the JD persistently ===
JD_FILE = "stored_jd.txt"

# === Load stored JD if exists ===
def load_stored_jd():
    if os.path.exists(JD_FILE):
        with open(JD_FILE, "r", encoding="utf-8") as file:
            return file.read()
    return ""

# === Save new JD to file ===
def save_jd(jd_text):
    with open(JD_FILE, "w", encoding="utf-8") as file:
        file.write(jd_text)

# === Gemini Prompt for Resume Analysis ===
def get_gemini_response(jd, resume_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a skilled ATS (Applicant Tracking System) with deep understanding of tech fields like software engineering, data engineering, data analysis, quantitative development, project management, and full-stack development.

    Your task is to evaluate resumes based on the given job description. The job market is competitive, so give best suggestions for improvement.

    Compare the following job description and resume. Give a match percentage and list missing keywords.

    Job Description:
    {jd}

    Resume:
    {resume_text}

    Respond ONLY in this JSON format:
    {{
      "match_percentage": "...",
      "missing_keywords": ["...", "..."]
    }}
    """
    response = model.generate_content(prompt)
    return response.text

# === Gemini Prompt for Unique Projects/Experience ===
def extract_unique_projects(resume_text, jd):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a skilled talent acquisition specialist. Extract the most unique or impressive projects, experiences, or skills from this resume that are relevant to the job description.
    
    Focus on identifying 1-3 standout elements that make this candidate special. These should be specific projects, unique experiences, or rare combinations of skills - not generic statements.
    
    Job Description:
    {jd}
    
    Resume:
    {resume_text}
    
    Return ONLY a concise JSON in this format:
    {{
      "unique_highlights": ["Project/Experience 1", "Project/Experience 2", "Project/Experience 3"]
    }}
    
    Keep each highlight under 100 characters. Focus on quality over quantity. If there are no truly unique elements, include the most relevant experiences instead.
    """
    response = model.generate_content(prompt)
    return response.text

# === Extract Text from PDF ===
def input_pdf_text(file_content):
    reader = pdf.PdfReader(file_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# === Process ZIP file containing PDFs ===
def process_zip_file(zip_file):
    extracted_files = []
    
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.lower().endswith('.pdf'):
                with z.open(filename) as file:
                    # Read file content
                    content = io.BytesIO(file.read())
                    extracted_files.append((os.path.basename(filename), content))
    
    return extracted_files

# === Clean percentage string to numeric value ===
def clean_percentage(percent_str):
    if isinstance(percent_str, str):
        match = re.search(r'(\d+)', percent_str)
        if match:
            return int(match.group(1))
    return 0

# === Create charts for analysis ===
def create_charts(results_df):
    # Convert percentage strings to numeric values for analysis
    results_df['Numeric Match'] = results_df['Match %'].apply(clean_percentage)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart - distribution of match ranges
    ranges = [
        '0-25%', 
        '26-50%', 
        '51-75%', 
        '76-90%', 
        '91-100%'
    ]
    
    # Count resumes in each range
    counts = [
        sum((results_df['Numeric Match'] >= 0) & (results_df['Numeric Match'] <= 25)),
        sum((results_df['Numeric Match'] > 25) & (results_df['Numeric Match'] <= 50)),
        sum((results_df['Numeric Match'] > 50) & (results_df['Numeric Match'] <= 75)),
        sum((results_df['Numeric Match'] > 75) & (results_df['Numeric Match'] <= 90)),
        sum((results_df['Numeric Match'] > 90) & (results_df['Numeric Match'] <= 100))
    ]
    
    # Filter out zeros to avoid empty slices
    non_zero_ranges = [ranges[i] for i in range(len(counts)) if counts[i] > 0]
    non_zero_counts = [counts[i] for i in range(len(counts)) if counts[i] > 0]
    
    if non_zero_counts:  # Only create pie if we have data
        # Create pie chart
        ax1.pie(non_zero_counts, labels=non_zero_ranges, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Resume Match Percentages')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    else:
        ax1.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
    
    # Histogram - frequency of match percentages
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax2.hist(results_df['Numeric Match'], bins=bins, color='skyblue', edgecolor='black')
    ax2.set_title('Frequency of Match Percentages')
    ax2.set_xlabel('Match Percentage')
    ax2.set_ylabel('Number of Resumes')
    ax2.set_xticks(bins)
    
    plt.tight_layout()
    return fig

# === Streamlit UI ===
st.title("Enhanced ATS Matcher")
st.markdown("**Improve your resume match with job descriptions and find top candidates**")

# === Sidebar for Settings ===
st.sidebar.header("Settings")
match_threshold = st.sidebar.slider("High Match Threshold (%)", 50, 100, 75)
show_analytics = st.sidebar.checkbox("Show Analytics Dashboard", True)

# === Main Content ===
stored_jd = load_stored_jd()
jd = st.text_area("Paste the Job Description", value=stored_jd, height=250)

if st.button("Save Job Description"):
    save_jd(jd)
    st.success("Job Description saved for future use!")

# Tab layout for different upload methods
upload_tab, zip_tab = st.tabs(["Upload Individual PDFs", "Upload Folder (ZIP)"])

with upload_tab:
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True, 
                                    help="Upload one or more PDF resumes")
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} individual PDF files")

with zip_tab:
    st.write("For folder upload: Zip your folder of PDFs and upload the ZIP file")
    zip_file = st.file_uploader("Upload ZIP file containing PDFs", type="zip")
    
    if zip_file:
        with st.spinner("Processing ZIP file..."):
            extracted_files = process_zip_file(zip_file)
            st.write(f"Found {len(extracted_files)} PDF files in the ZIP archive")

submit = st.button("Analyze Resumes")

if submit and jd:
    results = []
    raw_resume_texts = {}  # Store resume texts for further analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total files to process
    total_files = len(uploaded_files) if uploaded_files else 0
    if 'zip_file' in locals() and zip_file is not None:
        total_files += len(extracted_files)
    
    if total_files == 0:
        st.warning("No files to process. Please upload PDF files.")
    else:
        st.info(f"Processing {total_files} resumes. This may take some time...")
        files_processed = 0
        
        # Process individual uploaded PDFs
        if uploaded_files:
            for file in uploaded_files:
                try:
                    resume_text = input_pdf_text(file)
                    file_name = str(getattr(file, "name", "Unknown")).strip()
                    raw_resume_texts[file_name] = resume_text
                    
                    status_text.text(f"Analyzing {file_name}...")
                    result_text = get_gemini_response(jd, resume_text)
                    
                    # Handle possible markdown formatting or invalid JSON
                    result_text = result_text.strip().strip("```json").strip("```").strip()
                    
                    result_json = json.loads(result_text)
                    match_percentage = result_json.get("match_percentage", "N/A")
                    missing_keywords = result_json.get("missing_keywords", [])
                    
                    # Convert missing keywords list to string for display
                    missing_keywords_str = ", ".join(missing_keywords)
                    
                    results.append({
                        "Filename": file_name,
                        "Match %": str(match_percentage),
                        "Missing Keywords": missing_keywords_str,
                        "Raw Match": clean_percentage(str(match_percentage))
                    })
                except Exception as e:
                    st.error(f"Error processing {getattr(file, 'name', 'Unknown')}: {str(e)}")
                
                files_processed += 1
                progress_bar.progress(files_processed / total_files)
        
        # Process files from ZIP upload
        if 'zip_file' in locals() and zip_file is not None:
            for file_name, file_content in extracted_files:
                try:
                    resume_text = input_pdf_text(file_content)
                    raw_resume_texts[file_name] = resume_text
                    
                    status_text.text(f"Analyzing {file_name}...")
                    result_text = get_gemini_response(jd, resume_text)
                    
                    # Handle possible markdown formatting or invalid JSON
                    result_text = result_text.strip().strip("```json").strip("```").strip()
                    
                    result_json = json.loads(result_text)
                    match_percentage = result_json.get("match_percentage", "N/A")
                    missing_keywords = result_json.get("missing_keywords", [])
                    
                    # Convert missing keywords list to string for display
                    missing_keywords_str = ", ".join(missing_keywords)
                    
                    results.append({
                        "Filename": file_name,
                        "Match %": str(match_percentage),
                        "Missing Keywords": missing_keywords_str,
                        "Raw Match": clean_percentage(str(match_percentage))
                    })
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                
                files_processed += 1
                progress_bar.progress(files_processed / total_files)
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            # Convert to DataFrame for easier manipulation
            results_df = pd.DataFrame(results)
            
            # Sort by match percentage (descending)
            results_df = results_df.sort_values(by="Raw Match", ascending=False)
            
            # Add unique projects/experience for high matches
            high_matches = results_df[results_df["Raw Match"] >= match_threshold]
            
            if not high_matches.empty:
                status_text.text("Extracting unique highlights from top candidates...")
                
                # Process only high-matching resumes to extract unique points
                for index, row in high_matches.iterrows():
                    filename = row["Filename"]
                    if filename in raw_resume_texts:
                        try:
                            unique_text = extract_unique_projects(raw_resume_texts[filename], jd)
                            unique_text = unique_text.strip().strip("```json").strip("```").strip()
                            unique_json = json.loads(unique_text)
                            highlights = unique_json.get("unique_highlights", [])
                            results_df.at[index, "Unique Highlights"] = " | ".join(highlights)
                        except Exception as e:
                            results_df.at[index, "Unique Highlights"] = "Error extracting highlights"
                
                status_text.empty()
            
            # === Display Analytics Dashboard ===
            if show_analytics:
                st.header("Resume Analysis Dashboard")
                
                # Create metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Resumes", len(results_df))
                with col2:
                    st.metric(f"High Matches (≥{match_threshold}%)", len(results_df[results_df["Raw Match"] >= match_threshold]))
                with col3:
                    if not results_df.empty:
                        average_match = round(results_df["Raw Match"].mean(), 1)
                        st.metric("Average Match %", f"{average_match}%")
                
                # Display charts
                st.subheader("Visual Analysis")
                fig = create_charts(results_df)
                st.pyplot(fig)
            
            # === Display Results Tables ===
            # First show high matches
            high_matches = results_df[results_df["Raw Match"] >= match_threshold].copy()
            
            if not high_matches.empty:
                st.header(f"🌟 Top Candidates (≥{match_threshold}% Match)")
                
                for i, (index, row) in enumerate(high_matches.iterrows()):
                    with st.container():
                        cols = st.columns([2, 1, 7])
                        cols[0].markdown(f"**{row['Filename']}**")
                        cols[1].markdown(f"**{row['Match %']}**")
                        unique_text = row.get('Unique Highlights', '')
                        if unique_text:
                            cols[2].markdown(f"**Unique Highlights:** {unique_text}")
                    
                    with st.expander("Show Details"):
                        st.markdown(f"**Missing Keywords:** {row['Missing Keywords']}")
                    
                    st.divider()
            else:
                st.warning(f"No resumes matched the {match_threshold}% threshold. Consider lowering the threshold.")
            
            # Then show all results
            st.header("All Resume Results")
            
            # Filter controls
            col1, col2 = st.columns([1, 3])
            with col1:
                show_all = st.checkbox("Show All Resumes", value=False)
            
            if show_all:
                display_df = results_df
            else:
                # Show top 10 by default
                display_df = results_df.head(10)
                st.info("Showing top 10 matches. Check 'Show All Resumes' to see all results.")
            
            # Create a clean display version without the Raw Match column
            display_clean = display_df.drop(columns=["Raw Match"])
            
            # Use Streamlit's native table display
            st.table(display_clean)
            
            # Add option to download results as CSV
            csv = results_df.drop(columns=["Raw Match"]).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Results as CSV",
                data=csv,
                file_name="ats_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No results were generated. There might be an issue with the analysis process.")