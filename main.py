import os
import base64
import traceback
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import pipeline
from flask import Flask, request, jsonify
from llama_cloud_services import LlamaParse

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

MAX_SIZE = 20 * 1024 * 1024

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_ai_or_human(text):
    labels = ["AI-Generated", "Human-Written"]
    results = classifier(text, candidate_labels=labels)
    return results["labels"][0], float(results["scores"][0])

@app.route('/')
def index():
    return '''
    <h1>Upload a PPT or PPTX file</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".ppt,.pptx,.pdf,.png,.jpg,.jpeg" required>
        <button type="submit">Upload</button>
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400
    allowed_ext = ('.ppt', '.pptx', '.pdf', '.png', '.jpg', '.jpeg')
    if not (file and file.filename.lower().endswith(allowed_ext)):
        return jsonify({'error': 'Invalid file type. Only PPT, PPTX, PDF, PNG, JPG, or JPEG allowed.'}), 400
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > MAX_SIZE:
        return jsonify({'error': 'File too large. Max size is 20MB.'}), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': 'Failed to save file.', 'details': str(e)}), 500
    
    api_key = LLAMA_CLOUD_API_KEY

    if not api_key:
        return jsonify({'error': 'LlamaParse API key not set in environment.'}), 500
    
    parser = LlamaParse(
        api_key=api_key,
        num_workers=4,
        verbose=True,
        language="en",
    )

    try:
        result = parser.parse(filepath)
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            image_documents = result.get_image_documents(
                include_screenshot_images=True,
                include_object_images=False,
                image_download_dir="./images",
            )

            images_info = []
            for img_doc in image_documents:
                img_b64 = base64.b64encode(img_doc.image_bytes).decode('utf-8')
                images_info.append({
                    'image_base64': img_b64,
                    'description': getattr(img_doc, 'description', ''),
                })

            if not images_info:
                with open(filepath, 'rb') as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                images_info.append({
                    'image_base64': img_b64,
                    'description': 'Uploaded image (no description from parser)',
                })

            pages = []
            for page in result.pages:
                pages.append({
                    'text': getattr(page, 'text', ''),
                    'md': getattr(page, 'md', ''),
                    'images': getattr(page, 'images', []),
                    'layout': getattr(page, 'layout', {}),
                    'structuredData': getattr(page, 'structuredData', {}),
                })
            return jsonify({
                'message': 'Image uploaded and parsed successfully', 
                'images': images_info, 
                'pages': pages
            }), 200
        
        else:
            markdown_documents = result.get_markdown_documents(split_by_page=True)
            markdown_strings = [doc.markdown if hasattr(doc, 'markdown') else str(doc) for doc in markdown_documents]
            return jsonify({
                'message': 'File uploaded and parsed successfully',
                'markdown_documents': markdown_strings
            }), 200
    
    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()
        print("--- Exception Traceback ---")
        print(tb)

        if 'DNS resolution failed' in err_msg or 'Name or service not known' in err_msg:
            return jsonify({
                'error': 'DNS resolution failed. Cannot reach LlamaParse API.',
                'suggestions': [
                    'Check your internet connection',
                    'Try changing DNS to 8.8.8.8 or 1.1.1.1',
                    'Flush DNS cache: ipconfig /flushdns',
                    'Check firewall settings'
                ],
                'traceback': tb
            }), 500
        
        if '401' in err_msg or 'Unauthorized' in err_msg or 'Invalid token format' in err_msg:
            return jsonify({
                'error': 'LlamaParse API key is invalid or expired.',
                'suggestions': [
                    'Check your API key',
                    'Generate a new API key from LlamaIndex Cloud',
                    'Set the correct key in your code or environment variable'
                ],
                'traceback': tb
            }), 401
        
        return jsonify({'error': 'LlamaParse Python client failed', 'py_error': err_msg, 'traceback': tb}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client['sih-reg']
        teams_collection = db['teams']
        ps_collection = db['problemstatements']

        teams_data = []
        for team in teams_collection.find({}, { 'teamName': 1, 'tasks.files': 1, 'problemStatement': 1, '_id': 0 }):
            ppt_links = []
            if 'tasks' in team:
                for task in team['tasks']:
                    if 'files' in task and task['files']:
                        ppt_links.extend(task['files'])

            # Return the object that is two down from teamName (problemStatement as string if ObjectId)
            from bson import ObjectId
            ps = team.get('problemStatement')
            ps_id = None
            if isinstance(ps, ObjectId):
                ps_id = ps
                ps = str(ps)
            elif isinstance(ps, dict) and '$oid' in ps:
                ps_id = ObjectId(ps['$oid'])
                ps = ps['$oid']
            elif isinstance(ps, str):
                try:
                    ps_id = ObjectId(ps)
                except Exception:
                    ps_id = None

            # Fetch the title from problemstatements collection
            ps_title = None
            ps_description = None
            if ps_id:
                ps_obj = ps_collection.find_one({'_id': ps_id})
                if ps_obj:
                    if 'title' in ps_obj:
                        ps_title = ps_obj['title']
                    if 'description' in ps_obj:
                        ps_description = ps_obj['description']

            team_data = {
                'teamName': team.get('teamName'),
                'pptLinks': ppt_links,
                'problemStatement': ps,
                'psTitle': ps_title,
                'psDescription': ps_description
            }
            teams_data.append(team_data)

        client.close()
        return jsonify(teams_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/clean_uploads', methods=['DELETE'])
def clean_uploads():
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({"message": "Uploads folder does not exist"}), 404

        deleted_files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files.append(filename)
            except Exception as e:
                return jsonify({"error": f"Error deleting {filename}: {str(e)}"}), 500

        return jsonify({
            "message": "Uploads folder cleaned successfully",
            "deleted_files": deleted_files
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Invalid request. Please provide 'text' field."}), 400

    text = data["text"]
    label, score = detect_ai_or_human(text)

    score = int(score * 10)

    return jsonify({
        "label": label,
        "score": score
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)