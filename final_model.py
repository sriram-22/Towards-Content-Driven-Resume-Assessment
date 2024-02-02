import docx
import magic
import os
import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk
nltk.data.path.append('/path/to/your/nltk_data')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

#Extract Text from files
def extract_text_from_pdf(pdf_path):
    text = ''
    with fitz.open(pdf_path) as pdf_doc:
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text(file_path):
    mime = magic.Magic()
    file_type = mime.from_file(file_path)

    if 'PDF' in file_type:
        return extract_text_from_pdf(file_path)
    elif 'Microsoft Word' in file_type:
        return extract_text_from_docx(file_path)
    else:
        return 'Unsupported file format'

#clean Text data
def clean_text(text):
  '''
  def remove_special_characters(text):
      return re.sub(r'[^a-zA-Z0-9\s]', '', text)
  '''
  def convert_to_lowercase(text):
      return text.lower()



  def remove_stop_words(text):
      stop_words = set(stopwords.words('english'))
      words = word_tokenize(text)
      filtered_text = [word for word in words if word.lower() not in stop_words]
      return ' '.join(filtered_text)


  def lemmatize_text(text):
      lemmatizer = WordNetLemmatizer()
      words = word_tokenize(text)
      lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
      return ' '.join(lemmatized_text)


  #cleaned_text = remove_special_characters(text)
  cleaned_text = convert_to_lowercase(text)
  cleaned_text = remove_stop_words(cleaned_text)
  cleaned_text = lemmatize_text(cleaned_text)
  return cleaned_text


#TF-IDF Score
def score(resume_text,job_text):
  if not resume_text.strip() or not job_text.strip():
      print("No non-empty content to process.")
  else:
      # Combine the texts for vectorization
      content = [resume_text, job_text]

      # Define a custom tokenizer function using NLTK
      def custom_tokenizer(text):
          # Your custom tokenization logic here
          tokens = word_tokenize(text)
          tokens = [re.sub('[^A-Za-z]', '', token).lower() for token in tokens]
          return tokens

      # Create TfidfVectorizer with custom tokenizer
      tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

      # Fit and transform the data
      tfidf_matrix = tfidf_vectorizer.fit_transform(content)

      # Calculate cosine similarity
      similarity_matrix = cosine_similarity(tfidf_matrix)
      #print("Cosine Similarity Matrix:")
      #print(similarity_matrix)
      score = similarity_matrix[0, 1]
      return score



def perform_lda(resume, job):
    content = [resume, job]

    # Vectorize the content using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(content)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    document_topics = lda.transform(X)
    jaccard_similarity = 1 - np.abs(document_topics[0] - document_topics[1]).sum()

    # Convert the Jaccard similarity to a range of [0, 1]
    normalized_similarity = (jaccard_similarity + 1) / 2

    return normalized_similarity





def final_model(resume_path,job_path):
    resume_text = extract_text(resume_path)
    job_text = extract_text(job_path)


    cleaned_resume_text = clean_text(resume_text)
    cleaned_job_text = clean_text(job_text)

    lda_similarity = perform_lda(resume_text, job_text)
    tf_idf_Score = score(cleaned_resume_text,cleaned_job_text)
    #return (lda_similarity*100)
    return round(((lda_similarity*0.2+tf_idf_Score*0.8)*100+100)/2,2)

'''
resume_path = 'D:/FOLDER/final year project/project/Major Project/resources/Jobs/A career in our Cybersecurity.docx'
job_path = 'D:/FOLDER/final year project/project/Major Project/resources/Resumes/sample_resume1.pdf'
print(final_model(resume_path,job_path))
'''

def score_all_combinations(resume_folder, job_desc_folder):
    # Get the list of resume and job description files
    resume_files = os.listdir(resume_folder)
    job_desc_files = os.listdir(job_desc_folder)
    scores = []
    # Iterate through each combination of resume and job description files
    for resume_file in resume_files:
        for job_desc_file in job_desc_files:
            # Construct the full paths for each file
            resume_path = os.path.join(resume_folder, resume_file)
            job_desc_path = os.path.join(job_desc_folder, job_desc_file)

            # Calculate the score for the current combination
            
            scores.append(final_model(resume_path, job_desc_path))

    return sum(scores)//len(scores)
    

# Specify the paths to your resume and job description folders
resume_folder_path = "resources/INFORMATION-TECHNOLOGY"
job_desc_folder_path = "resources/Jobs"

print('Total Score: ',score_all_combinations(resume_folder_path, job_desc_folder_path))
