from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: list of resumes
resumes = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]

# Step 2: job description
job_description = "python java sql machine learning"

results = []

for file in resumes:
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    resume_text = text.lower()

    # TF-IDF
    documents = [job_description, resume_text]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    results.append((file, similarity))

# Step 3: sort results
results.sort(key=lambda x: x[1], reverse=True)

# Step 4: print ranking
for r in results:
    print(r[0], "→", round(r[1]*100, 2), "%")
