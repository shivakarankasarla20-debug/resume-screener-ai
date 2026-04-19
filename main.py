from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

reader = PdfReader("resume.pdf")
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

resume_text = text.lower()

job_description = "python java sql machine learning"

documents = [job_description, resume_text]

cv = CountVectorizer()
matrix = cv.fit_transform(documents)

print("Words:", cv.get_feature_names_out())
print("Matrix:\n", matrix.toarray())

similarity = cosine_similarity(matrix)[0][1]

print("Similarity Score:", similarity)
