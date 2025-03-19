from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os
import pdfplumber
import uuid

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Set a secret key for session management
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Ensure directories exist
os.makedirs('./uploads', exist_ok=True)

# OpenRouter API Key (Replace with your key)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Function to process PDFs and extract text
def process_pdf(pdf_files):
    extracted_text = []
    
    try:
        for pdf_file in pdf_files:
            if not pdf_file.filename.endswith('.pdf'):
                return None, "Invalid file type. Please upload a PDF."

            file_path = os.path.join('./uploads', pdf_file.filename)
            pdf_file.save(file_path)

            print(f"Processing PDF: {file_path}")

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text.append(text)

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        print(f"Table: {table}")
                        extracted_text.append(str(table))

            if not extracted_text:
                return None, "No text extracted from the PDF. Please upload a valid PDF."

        extracted_text = " ".join(extracted_text)
        print(f"Final Extracted Text: {extracted_text}")
        return extracted_text, None
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return None, str(e)

# Function to call OpenRouter API and generate an answer
def get_deepseek_response(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }

    print(f"Payload to OpenRouter: {payload}")

    response = requests.post(API_URL, json=payload, headers=headers)

    print(f"Response from OpenRouter: {response.json()}")

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.json()}"

# Route to serve UI
@app.route('/')
def index():
    return render_template('index.html')

# API route to upload and process PDF
@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf_files' not in request.files or not request.files.getlist('pdf_files'):
            return jsonify({"error": "No PDF file uploaded. Please upload a file and try again."}), 400

        pdf_files = request.files.getlist('pdf_files')

        # Generate a unique session ID for the user if it doesn't exist
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())

        # Process the PDF
        extracted_text, error = process_pdf(pdf_files)

        if error:
            return jsonify({"error": error}), 400

        # Store the extracted text in the session
        session['extracted_text'] = extracted_text
        print("PDF uploaded and text extracted successfully.")

        return jsonify({"message": "PDF uploaded and processed successfully."})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# API route to ask questions based on the uploaded PDF
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        if 'extracted_text' not in session:
            print("Error: No text extracted from PDF. Please upload a valid PDF.")
            return jsonify({"error": "No text extracted from PDF. Please upload a valid PDF."}), 400

        question = request.form.get('question', '').strip()

        if not question:
            print("Error: No question provided.")
            return jsonify({"error": "Please provide a question."}), 400

        extracted_text = session['extracted_text']
        print(f"Extracted Text: {extracted_text}")
        print(f"Question: {question}")

        answer = get_deepseek_response(question, extracted_text)

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port)