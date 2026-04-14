# utils.py
import re
import pandas as pd
import numpy as np
import joblib
import spacy
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
from datetime import datetime, timezone
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"

nlp = spacy.load("en_core_web_sm")
# Use a lightweight but good model (~80-100 MB RAM usage)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ====================== CRITICAL: CUSTOM TOKENIZER FOR MODEL LOADING ======================
import re

def skill_tokenizer(text):
    """Custom tokenizer used when the XGBoost model was trained.
    Must be defined exactly with this name and at module level."""
    if not isinstance(text, str) or not text.strip():
        return []
    # Split on commas, semicolons, newlines
    tokens = re.split(r'[,\n;]', text)
    # Clean and lower
    skills = [tok.strip().lower() for tok in tokens if tok.strip()]
    return skills


# Make sure it's available in the global namespace
__all__ = ['skill_tokenizer']

# ==================== PDF & Text Extraction ====================
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|github\.com/\S+|linkedin\.com/in/\S+)", re.IGNORECASE)
TRAILING_JUNK = ")]}>.,;:\"'"

def clean_url(url):
    url = url.strip().strip(TRAILING_JUNK)
    if url.startswith("www."):
        url = "https://" + url
    if url.startswith(("github.com", "linkedin.com")):
        url = "https://" + url
    return url.lower()

def ocr_page(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)

def extract_text_and_links(pdf_path):
    texts = []
    links = set()

    # Extract text using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2)
            if txt and len(txt.strip()) > 30:
                texts.append(txt)

    # Use PyMuPDF for links and OCR fallback
    doc = fitz.open(pdf_path)
    for page in doc:
        if not page.get_text().strip():
            texts.append(ocr_page(page))
        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                links.add(clean_url(uri))
        for url in URL_PATTERN.findall(page.get_text()):
            links.add(clean_url(url))
    doc.close()

    final_text = "\n".join(dict.fromkeys(texts))
    return final_text.strip(), sorted(links)

# ==================== Skill Extraction (NLP) ====================
EXPLICIT_SKILLS = [
    "python", "java", "c", "c++", "javascript", "typescript", "go", "rust", "ruby", "php", "kotlin", "swift",
    "html", "css", "react", "angular", "vue", "next js", "bootstrap", "tailwind", "node js", "express",
    "django", "flask", "fastapi", "sql", "mysql", "postgresql", "mongodb", "redis", "aws", "azure", "gcp",
    "docker", "kubernetes", "git", "github", "postman", "swagger"
]

SKILL_SYNONYMS = {
    "js": "javascript", "py": "python", "postgres": "postgresql",
    "node": "node js", "tf": "tensorflow", "ml": "machine learning"
}

IMPLICIT_SKILL_CONCEPTS = [
    "machine learning", "deep learning", "data science", "natural language processing",
    "computer vision", "backend development", "frontend development", "cloud computing",
    "data structures and algorithms", "system design"
]

def extract_skills_nlp(text):
    detected_skills = {}
    text_lower = text.lower()
    text_clean = re.sub(r'[^a-z0-9\s+#]', ' ', text_lower)

    for skill in EXPLICIT_SKILLS:
        if re.search(rf"\b{re.escape(skill)}(\s+|$)", text_clean):
            detected_skills[skill] = 0.98

    for key, value in SKILL_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", text_clean):
            detected_skills[value] = max(detected_skills.get(value, 0), 0.95)

    try:
        doc = nlp(text_lower)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        if sentences:
            sent_embeddings = embedder.encode(sentences, convert_to_tensor=True)
            concept_embeddings = embedder.encode(IMPLICIT_SKILL_CONCEPTS, convert_to_tensor=True)
            cosine_scores = util.cos_sim(concept_embeddings, sent_embeddings)
            for j, concept in enumerate(IMPLICIT_SKILL_CONCEPTS):
                max_score = float(cosine_scores[j].max())
                threshold = 0.80 if "development" in concept else 0.85
                if max_score >= threshold:
                    detected_skills[concept] = max(detected_skills.get(concept, 0), max_score)
    except Exception as e:
        print("NLP Error:", e)

    skills_sorted = sorted(detected_skills.keys(), key=lambda x: (-detected_skills[x], x))
    return ", ".join(skills_sorted)

# ==================== Other Feature Extractors ====================
def extract_projects(text):
    if not text or len(text) < 80:
        return 0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    PROJECT_HEADERS = {"projects", "project", "academic projects", "technical projects", "key projects"}
    STOP_HEADERS = {"experience", "education", "skills", "certifications", "internship", "achievements"}
    ACTION_WORDS = {"developed","implemented","designed","built","created","engineered","performed","enabled",
                    "integrated","used","worked","applied","analyzed","trained","deployed"}

    def looks_like_title(line):
        words = line.split()
        return (not line.startswith(("•","-","–")) and 2 <= len(words) <= 8 and words[0].lower() not in ACTION_WORDS)

    start = -1
    for i, line in enumerate(lines):
        if line.lower() in PROJECT_HEADERS:
            start = i + 1
            break
    if start == -1:
        return 0

    titles = []
    i = start
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        if low in STOP_HEADERS:
            break
        if looks_like_title(line):
            if i + 1 >= len(lines):
                i += 1
                continue
            next_line = lines[i + 1]
            is_description = (next_line.startswith(("•","-","–")) or 
                             next_line.split()[0].lower() in ACTION_WORDS or 
                             len(next_line.split()) > 8 or ":" in next_line)
            is_followed_by_title = (i + 2 < len(lines) and looks_like_title(lines[i + 2]))
            if is_description and not is_followed_by_title:
                titles.append(low)
                i += 1
                continue
        i += 1
    return len(set(titles))

def extract_certifications_nlp(text):
    CERT_HEADERS = {"certification", "certifications", "certificate", "certificates"}
    STOP_HEADERS = {"projects", "experience", "education", "skills", "internship", "achievements"}
    EXCLUDE_TERMS = {"hackathon", "workshop", "participation", "organized", "event", "internship"}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    start, end = -1, len(lines)
    for i, line in enumerate(lines):
        if line.lower() in CERT_HEADERS:
            start = i + 1
            break
    if start == -1:
        return 0
    for j in range(start, len(lines)):
        if lines[j].lower() in STOP_HEADERS:
            end = j
            break

    certifications = []
    for line in lines[start:end]:
        low = line.lower()
        if any(x in low for x in EXCLUDE_TERMS):
            continue
        if len(low.split()) < 3:
            continue
        doc = nlp(line)
        if any(tok.dep_ == "nsubj" for tok in doc):
            continue
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
        if len(nouns) < 2:
            continue
        clean = re.sub(r"\s+", " ", low)
        if not any(re.sub(r"\s+", " ", c).strip() == clean for c in certifications):  # simplified duplicate check
            certifications.append(clean)
    return len(certifications)

def extract_hackathons_nlp(text):
    doc = nlp(text.lower())
    count = 0
    for sent in doc.sents:
        if any(k in sent.text for k in ["hackathon", "datathon", "codefest", "ideathon", "flowthon"]) and \
           any(w in sent.text for w in ["participated", "won", "developed", "built"]):
            count += 1
    return count

def extract_coding_problems(text: str) -> int:
    patterns = [
        r'solved\s+(\d+\+?)\s+problems',
        r'(\d+\+?)\s+coding\s+problems',
        r'(\d+\+?)\s+problems\s+solved',
        r'solved\s+(\d+\+?)'
    ]
    combined_pattern = '|'.join(patterns)
    total_count = 0
    for match in re.finditer(combined_pattern, text, re.IGNORECASE):
        for group in match.groups():
            if group:
                try:
                    total_count += int(group.replace('+', ''))
                except ValueError:
                    continue
                break
    return total_count

def extract_cgpa(text):
    text = text.lower()
    patterns = [
        r'(?:cgpa|gpa|cumulative gpa|c\.g\.p\.a)\s*[:\-]?\s*(\d\.\d{1,2})\s*/?\s*10?',
        r'(\d\.\d{1,2})\s*(?:cgpa|gpa)',
        r'(\d\.\d{1,2})\s*(?:out of)\s*10',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            val = float(m.group(1))
            if 0.0 < val < 10.0:
                return val
    return 0.0

def extract_github_username(pdf_path):
    text, links = extract_text_and_links(pdf_path)
    raw = re.findall(r"github\.com/([A-Za-z0-9_-]+)", text, re.IGNORECASE)
    if raw:
        return raw[0].lower()
    for link in links:
        m = re.search(r"github\.com/([A-Za-z0-9_-]+)", link, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None

# ==================== GitHub Repo Scoring ====================
def github_get(url):
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 403:
            print("GitHub API Rate Limit Hit")
            return None
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def extract_repo_metrics(owner, repo):
    repo_url = f"{GITHUB_API}/repos/{owner}/{repo}"
    data = github_get(repo_url)
    if not data:
        return None

    stars = data.get("stargazers_count", 0)
    forks = data.get("forks_count", 0)
    open_issues = data.get("open_issues_count", 0)

    pushed_at = data.get("pushed_at")

    if pushed_at:
        days_since_update = (
            datetime.now(timezone.utc) -
            datetime.strptime(pushed_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
     ).days
    else:
        days_since_update = 9999
    freshness = max(0, 1 - min(days_since_update, 365) / 365)

    contrib_data = github_get(f"{repo_url}/contributors?per_page=30")
    contributors = len(contrib_data) if isinstance(contrib_data, list) else 1

    pr_data = github_get(f"{repo_url}/pulls?state=all&per_page=30")
    prs = len(pr_data) if isinstance(pr_data, list) else 0

    closed_data = github_get(f"{GITHUB_API}/search/issues?q=repo:{owner}/{repo}+type:issue+state:closed")
    closed_issues = closed_data.get("total_count", 0) if isinstance(closed_data, dict) else 0
    issue_health = closed_issues / (closed_issues + open_issues) if (closed_issues + open_issues) > 0 else 0

    release_data = github_get(f"{repo_url}/releases?per_page=10")
    releases = len(release_data) if isinstance(release_data, list) else 0

    return [stars, forks, contributors, prs, issue_health, freshness, releases]

def topsis(matrix, weights):
    matrix = np.array(matrix, dtype=float)
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1
    norm_matrix = matrix / norms
    weighted_matrix = norm_matrix * weights

    ideal_best_raw = np.array([3, 2, 1, 1, 1, 1, 1])
    ideal_worst_raw = np.array([0, 0, 0, 0, 0, 0, 0])

    ideal_best = (ideal_best_raw / norms) * weights
    ideal_worst = (ideal_worst_raw / norms) * weights

    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    epsilon = 1e-10
    scores = dist_worst / (dist_best + dist_worst + epsilon)
    return scores

def github_score(username):
    if not username:
        return 0, 0
    repos_url = f"{GITHUB_API}/users/{username}/repos?per_page=30"
    repos = github_get(repos_url)
    if not repos or not isinstance(repos, list) or len(repos) == 0:
        return 0, 0

    decision_matrix = []
    for repo in repos[:15]:
        owner = repo["owner"]["login"]
        name = repo["name"]
        metrics = extract_repo_metrics(owner, name)
        if metrics:
            decision_matrix.append(metrics)

    if len(decision_matrix) == 0:
        return 0, 0

    weights = np.array([0.25, 0.10, 0.15, 0.15, 0.15, 0.20, 0.10])
    weights = weights / weights.sum()

    repo_scores = topsis(decision_matrix, weights)
    profile_score = np.mean(repo_scores)
    final_score = round(profile_score * 100, 2)
    return len(repo_scores), final_score

def detect_linkedin(text, links):
    if any("linkedin.com" in l for l in links) or "linkedin" in text.lower():
        return 1
    return 0

def detect_professional_membership(text):
    text_lower = text.lower()
    if re.search(r"\b(ieee|iste|csi)\b", text_lower):
        return 1
    return 0

# ==================== Build Feature Row ====================
def build_feature_row_nlp(pdf_path):
    text, links = extract_text_and_links(pdf_path)
    github_user = extract_github_username(pdf_path)
    repos, repo_score = github_score(github_user)

    return pd.DataFrame({
        "Skills": [extract_skills_nlp(text)],
        "Projects": [extract_projects(text)],
        "Certifications": [extract_certifications_nlp(text)],
        "GitHub Repos": [repos],
        "GitHub Repo Score": [repo_score],
        "Hackathons": [extract_hackathons_nlp(text)],
        "Coding Problems Solved": [extract_coding_problems(text)],
        "LinkedIn": [detect_linkedin(text, links)],
        "Professional Body Membership": [detect_professional_membership(text)],
        "CGPA": [extract_cgpa(text)]
    })

# ==================== Final Prediction ====================
def predict_resume_grade(pdf_path: str, model) -> str:
    X = build_feature_row_nlp(pdf_path)
    pred_encoded = model.predict(X)[0]
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return grade_map.get(pred_encoded, "D")