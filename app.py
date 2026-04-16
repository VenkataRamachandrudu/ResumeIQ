# ===== STEP 1: REGISTER TOKENIZER FIRST (CRITICAL) =====
import __main__
import re

def skill_tokenizer(text):
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = re.split(r'[,\n;]', text)
    return [tok.strip().lower() for tok in tokens if tok.strip()]

__main__.skill_tokenizer = skill_tokenizer


# ===== STEP 2: IMPORT LIBRARIES =====
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import joblib

# ===== STEP 3: LOAD MODEL AFTER TOKENIZER REGISTERED =====
model = joblib.load("resume_grading_xgb_mode.pkl")

# ===== STEP 4: IMPORT UTILS =====
from utils import predict_resume_grade

# ===== STEP 5: FLASK APP =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        grade = predict_resume_grade(filepath, model)
        return jsonify({"grade": grade})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)