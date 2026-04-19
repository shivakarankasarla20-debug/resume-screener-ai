from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read resume
reader = PdfReader("resume.pdf")
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

resume_text = text.lower()

# Step 2: Job description
job_description = "python java sql machine learning"

# Step 3: Convert text to numbers
documents = [job_description, resume_text]

cv = CountVectorizer()
matrix = cv.fit_transform(documents)

# 🔥 ADD THESE LINES (to understand conversion)
print("Words:", cv.get_feature_names_out())
print("Matrix:\n", matrix.toarray())

# Step 4: Calculate similarity
similarity = cosine_similarity(matrix)[0][1]

# Step 5: Output
print("Similarity Score:", similarity)