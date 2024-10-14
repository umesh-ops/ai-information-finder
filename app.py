import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the vectorizer and TF-IDF matrix
docs_path = r"C:\Users\umesh\Project Documents code"  # Update this line
with open(os.path.join(docs_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

with open(os.path.join(docs_path, 'tfidf_matrix.pkl'), 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(docs_path, 'doc_names.pkl'), 'rb') as f:
    doc_names = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    print(f"User Query: {query}")  # Debug: print the user query
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    print(f"Similarities: {similarities}")  # Debug: print similarities array
    
    most_similar_doc_indices = similarities.argsort()[::-1][:5]  # Get top 5 results

    results = [(doc_names[i], similarities[i]) for i in most_similar_doc_indices]

    print(f"Results: {results}")  # Debug: print the results
    
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

