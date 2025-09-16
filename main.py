import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from llama_cloud_services import LlamaParse
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()

LLAMA_API_KEY = os.environ.get('LLAMA_CLOUD_API_KEY')

if not LLAMA_API_KEY:
    raise RuntimeError('LLAMA_CLOUD_API_KEY environment variable not set')

@app.route('/parse', methods=['POST'])
def parse_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            num_workers=1,
            verbose=True,
            language="en"
        )
    result = parser.parse(file_path)
    text_documents = result.get_text_documents(split_by_page=False)
    text = "\n".join([doc.text for doc in text_documents])
    return jsonify({'text': text, 'embedding_saved': True})

if (__name__ == "__main__"):
    app.run(debug=True, port=5000, host="0.0.0.0")