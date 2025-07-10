from flask import Flask, render_template, request, send_file, jsonify
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
from PyPDF2 import PdfReader
from flask_cors import CORS
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from fpdf import FPDF
import json

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
Bootstrap(app)
CORS(app)

nlp = spacy.load("en_core_web_sm")

# --- Context-Aware MCQ Generation ---
def cluster_sentences(sentences, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    clusters = kmeans.labels_.tolist()

    clustered_sentences = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(clusters):
        clustered_sentences[label].append(sentences[i])

    return clustered_sentences

# --- Relevance Calculation ---
def calculate_relevance(question, sentence):
    vectorizer = TfidfVectorizer().fit_transform([question, sentence])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]  # Similarity score between question and sentence

def generate_mcqs(text, num_questions=5, question_type='mcq'):
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    num_questions = min(num_questions, len(sentences))
    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        if len(nouns) < 2:
            continue

        noun_counts = Counter(nouns)
        subject = noun_counts.most_common(1)[0][0]
        question_stem = sentence.replace(subject, "______")

        if question_type == 'mcq':
            answer_choices = [subject]
            distractors = list(set(nouns) - {subject})
            while len(distractors) < 3:
                distractors.append("[Distractor]")
            random.shuffle(distractors)
            answer_choices.extend(distractors[:3])
            random.shuffle(answer_choices)
            correct_answer = chr(64 + answer_choices.index(subject) + 1)

            # Calculate relevance score between sentence and question stem
            relevance_score = calculate_relevance(question_stem, sentence)

            mcqs.append((question_stem, answer_choices, correct_answer, relevance_score))

        elif question_type == 'true_false':
            answer = "True" if random.choice([True, False]) else "False"
            relevance_score = calculate_relevance(sentence, sentence)  # For true/false, same sentence
            mcqs.append((sentence, ["True", "False"], answer, relevance_score))

    return mcqs

def generate_contextual_mcqs(text, num_questions=5, num_clusters=3, question_type='mcq'):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    clustered_sentences = cluster_sentences(sentences, num_clusters=num_clusters)
    mcqs = []

    for cluster in clustered_sentences.values():
        selected_sentences = random.sample(cluster, min(num_questions, len(cluster)))
        mcqs.extend(generate_mcqs(' '.join(selected_sentences), num_questions=len(selected_sentences),
                                  question_type=question_type))

    return mcqs

# --- PDF Processing ---
def process_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            text += page_text
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
    return text

def generate_pdf(mcqs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for index, (question, choices, correct, relevance) in enumerate(mcqs):
        pdf.multi_cell(0, 10, f"{index + 1}. {question}")
        for choice in choices:
            pdf.cell(0, 10, f" - {choice}", ln=True)
        pdf.cell(0, 10, f"Relevance: {relevance:.2f}", ln=True)
        pdf.cell(0, 10, '', ln=True)  # Blank line

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
        else:
            text = request.form.get('text', '')

        try:
            num_questions = int(request.form.get('num_questions', 5))
        except ValueError:
            num_questions = 5

        question_type = request.form.get('question_type', 'mcq')
        mcqs = generate_contextual_mcqs(text, num_questions=num_questions, question_type=question_type)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]

        return render_template('mcqs.html', mcqs=mcqs_with_index, question_type=question_type)

    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download_mcqs():
    mcqs_json = request.form.get('mcqs')  # Get MCQs as JSON string
    if not mcqs_json:
        return jsonify({"error": "No MCQs provided"}), 400

    # Log the received JSON for debugging
    logging.debug(f"Received mcqs_json: {mcqs_json}")

    # Deserialize the JSON string back into a Python object
    try:
        mcqs = json.loads(mcqs_json)
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e}")
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    # Ensure the structure of mcqs is correct
    if not isinstance(mcqs, list):
        return jsonify({"error": "Invalid MCQs format"}), 400

    pdf_output = generate_pdf(mcqs)
    return send_file(pdf_output, as_attachment=True, download_name="mcqs.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
