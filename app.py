from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import os
import uuid
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Set a secret key for session management
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Ensure directories exist
os.makedirs('./uploads', exist_ok=True)
os.makedirs('./static/images', exist_ok=True)

# Load Hugging Face model and tokenizer
load_dotenv()  # Load environment variables from .env

model_name = "deepseek-ai/Janus-Pro-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to process PDFs and extract text, tables, and images
def process_pdf(pdf_files):
    extracted_text = []
    extracted_tables = []
    extracted_images = []
    
    try:
        for pdf_file in pdf_files:
            if not pdf_file.filename.endswith('.pdf'):
                return None, None, None, "Invalid file type. Please upload a PDF."

            file_path = os.path.join('./uploads', pdf_file.filename)
            pdf_file.save(file_path)

            print(f"Processing PDF: {file_path}")

            # Extract text and tables using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text.append(text)

                    tables = page.extract_tables()
                    for table in tables:
                        extracted_tables.append(table)

            # Extract images using PyMuPDF
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_name = f"image_{uuid.uuid4()}.{image_ext}"
                    image_path = os.path.join('./static/images', image_name)
                    with open(image_path, 'wb') as image_file:
                        image_file.write(image_bytes)
                    extracted_images.append(image_name)

            if not extracted_text and not extracted_tables and not extracted_images:
                return None, None, None, "No content extracted from the PDF. Please upload a valid PDF."

        extracted_text = " ".join(extracted_text)
        print(f"Final Extracted Text: {extracted_text}")
        return extracted_text, extracted_tables, extracted_images, None
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return None, None, None, str(e)

# Function to generate response using Janus-Pro-7B model
def get_janus_response(question, context):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate response
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=512,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        # Decode the generated tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()  # Extract the generated answer part

    except Exception as e:
        return f"Error: {str(e)}"

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
        extracted_text, extracted_tables, extracted_images, error = process_pdf(pdf_files)

        if error:
            return jsonify({"error": error}), 400

        # Store the extracted text, tables, and images in the session
        session['extracted_text'] = extracted_text
        session['extracted_tables'] = extracted_tables
        session['extracted_images'] = extracted_images
        print("PDF uploaded and content extracted successfully.")

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
        extracted_tables = session.get('extracted_tables', [])
        extracted_images = session.get('extracted_images', [])
        print(f"Extracted Text: {extracted_text}")
        print(f"Question: {question}")

        answer = get_janus_response(question, extracted_text)

        # Prepare response with answer, tables, and images
        response_data = {
            "answer": answer,
            "tables": extracted_tables,
            "images": [f"/static/images/{img}" for img in extracted_images]
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment, default to 5000
    host = os.environ.get("HOST", "0.0.0.0")  # Use HOST from environment, default to 0.0.0.0
    app.run(host=host, port=port)
