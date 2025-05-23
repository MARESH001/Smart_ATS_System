# Advanced ATS Matcher

A smart application that analyzes resumes against job descriptions using AI to help both job seekers and recruiters.

**[Try the live demo →](https://advanced-ats.streamlit.app/)**

## Features

- **For Job Seekers**: Get match percentages and identify missing keywords
- **For Recruiters**: Process multiple resumes, find top candidates, extract unique highlights
- **Analytics**: Visual dashboard showing match distributions and statistics

## Quick Start

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Google API key:
```
GOOGLE_KEY=your_google_api_key_here
```

3. Run the app:
```bash
streamlit run app.py
```

## How It Works

1. Enter a job description (can be saved for later)
2. Upload resumes (individual PDFs or ZIP folder)
3. Click "Analyze Resumes"
4. View top candidates and match percentages
5. Download results as CSV

## Tech Stack

- Streamlit
- Google Gemini 1.5 Flash
- PyPDF2
- Pandas and Matplotlib
