from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Ensure directories exist
os.makedirs('./uploads', exist_ok=True)

# OpenRouter API Key (Replace with your key)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global variable to store extracted text
extracted_text = None

# Function to process PDFs and extract text
def process_pdf(pdf_files):
    global extracted_text
    extracted_text = []
    
    try:
        for pdf_file in pdf_files:
            file_path = os.path.join('./uploads', pdf_file.filename)
            pdf_file.save(file_path)

            print(f"Processing PDF: {file_path}")

            loader = PyMuPDFLoader(file_path)
            data = loader.load()

            # Debug: Print the extracted data
            print(f"Extracted Data: {data}")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Debug: Print the chunks
            print(f"Chunks: {chunks}")

            for chunk in chunks:
                extracted_text.append(chunk.page_content)

        extracted_text = " ".join(extracted_text)  # Combine text from all chunks

        # Debug: Print the final extracted text
        print(f"Final Extracted Text: {extracted_text}")

        return True
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return False

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

    # Debug: Print the payload being sent to OpenRouter
    print(f"Payload to OpenRouter: {payload}")

    response = requests.post(API_URL, json=payload, headers=headers)

    # Debug: Print the response from OpenRouter
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
    global extracted_text
    try:
        if 'pdf_files' not in request.files or request.files.getlist('pdf_files') == []:
            return jsonify({"error": "No PDF files uploaded. Please upload a file and try again."}), 400

        pdf_files = request.files.getlist('pdf_files')

        # Process the PDF
        success = process_pdf(pdf_files)

        if not success:
            return jsonify({"error": "Failed to extract text from PDFs."}), 400

        return jsonify({"message": "PDF uploaded and processed successfully."})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# API route to ask questions based on the uploaded PDF
@app.route('/ask', methods=['POST'])
def ask_question():
    global extracted_text
    try:
        if extracted_text is None:
            return jsonify({"error": "No PDF has been uploaded yet. Please upload a PDF first."}), 400

        question = request.form.get('question', '').strip()

        if not question:
            return jsonify({"error": "Please provide a question."}), 400

        # Debug: Print the extracted text and question
        print(f"Extracted Text: {extracted_text}")
        print(f"Question: {question}")

        answer = get_deepseek_response(question, extracted_text)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    app.run(host="0.0.0.0", port=port)