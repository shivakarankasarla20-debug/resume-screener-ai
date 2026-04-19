from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]
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

    documents = [job_description, resume_text]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    results.append((file, similarity))

results.sort(key=lambda x: x[1], reverse=True)

for r in results:
    print(r[0], "→", round(r[1]*100, 2), "%")
