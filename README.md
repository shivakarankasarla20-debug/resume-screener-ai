# Resume Screener

I tried building a small project to understand how resume screening works.

This program reads a PDF resume and compares it with a job description.  
It then gives a similarity score based on how much the resume matches the required skills.

## What I used
Python, PyPDF2, scikit-learn

## How it works
- Reads resume from PDF
- Converts text into numbers
- Compares with job description
- Prints similarity score

## Run
pip install PyPDF2 scikit-learn  
python main.py

## Note
I used a sample (AI-generated) resume for testing.
