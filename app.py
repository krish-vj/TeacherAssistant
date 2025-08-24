import json
import logging
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import math
from google.cloud import documentai_v1 as documentai
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import re
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Document AI and Gemini configuration 
DOCUMENT_AI_PROJECT_ID = os.environ.get('DOCUMENT_AI_PROJECT_ID', 'yourProjectDir')
DOCUMENT_AI_LOCATION = os.environ.get('DOCUMENT_AI_LOCATION', 'replaceWIthYourStuff')
DOCUMENT_AI_PROCESSOR_ID = os.environ.get('DOCUMENT_AI_PROCESSOR_ID', 'replaceWithYourId')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')


if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment. Using fallback method.")
    # Fallback for development - REMOVE IN PRODUCTION
    GEMINI_API_KEY = "AIzaSyAXzQoCud0W3dooiXhliP0FdCGjjGj1yLo"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_ai_summary(text: str) -> str:
    """Get an AI-generated summary of the text."""
    try:
        prompt = f"""
        Summarize the following text, which may have structural issues due to PDF extraction.
        Focus on the main points and key information:
        
        {text}
        """
        response = model.generate_content(prompt)
        return process_markdown_response(response.text)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary. Please try again."

def get_ai_evaluation(text: str) -> str:
    """Get an AI-generated evaluation of the text based on specified criteria."""
    try:
        prompt = f"""
        The provided text may have a disorganized structure due to automatic extraction from a document using OCR or similar tools. 
        Assume that the structure is not critical for this evaluation 
        but focus on identifying issues such as:
        - Spelling mistakes (only major ones, as it's ocr extracted)
        - Major Grammatical errors (ignore minor)
        - Content quality and clarity, ignoring any lack of formatting or structure.
        
        Provide a short analysis and a grade out of 100. Format your response as follows:
        - **Overall Grade**: <numeric score>/<max_marks>
        - **number of question and answers **: <number of questions detected in text>
        - **Feedback**: Include short and sweet feedback on spelling, grammar, and content quality.
        - Format your response with proper Markdown:
        - Use ** for bold text
        - Use proper line breaks between sections
        - Use proper indentation for readability
        - Structure the response clearly with sections
        
        Text to analyze: {text}
        """
        response = model.generate_content(prompt)
        return process_markdown_response(response.text)
    except Exception as e:
        logger.error(f"Error generating evaluation: {str(e)}")
        return f"Error generating evaluation. Please try again."

def get_ai_evaluation_with_reference(student_text: str, reference_text: str) -> str:
    """Get an AI-generated evaluation of student text compared against reference text."""
    try:
        prompt = f"""
        You are grading a student's answer against an ideal answer. 
        KEEP the overall answer short and sweet!!
        First, understand the ideal answer to identify:
        1. See if Number of question and answer in ideal answer match! (Speicify if any question is missing)
        2. Key concepts that must be present
        3. The logical flow and structure expected
        
        
        
        Both texts were automatically extracted from documents and may have structural issues.
        Ignore minor formatting problems and focus on:
        - Content accuracy and completeness
        - Presence of key concepts
        - Understanding of the subject matter
        - Major spelling or grammatical errors
        
        Provide a grade out of 100 and clear feedback.
        Reduce the grade for missing questions. (Ex: ideal has 3 question answers, but student has only 2; so reduce marks)
        Format your response as follows:
        - **Grade**: <numeric score>/100
        - **Number of questions answered**: <number of questions answered>/<total number of questions detected>
        - **Strengths**: List main positive aspects
        - **Areas for Improvement**: Constructive feedback
        - **Overall Assessment**: Short summary justifying the grade
        
        IDEAL ANSWER:
        {reference_text}
        
        STUDENT ANSWER:
        {student_text}
        """
        
        response = model.generate_content(prompt)
        return process_markdown_response(response.text)
    except Exception as e:
        logger.error(f"Error generating evaluation with reference: {str(e)}")
        return f"Error generating evaluation. Please try again."

def process_markdown_response(text):
    # Replace Markdown bold syntax with HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace single asterisks for italic
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Convert double newlines to paragraph breaks
    text = re.sub(r'\n\n+', '</p><p>', text)
    
    # Convert single newlines to line breaks
    text = re.sub(r'\n', '<br>', text)
    
    # Wrap in paragraphs if not already wrapped
    if not text.startswith('<p>'):
        text = f'<p>{text}</p>'
    
    return text
def extract_document_text(file_path: str) -> Tuple[documentai.Document, List[str], List[Any]]:
    """
    Extract text and bounding boxes from a PDF using Google Document AI.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple containing:
        - Document AI document
        - List of words
        - List of bounding polygons
    """
    logger.info(f"Processing document: {file_path}")
    
    try:
        # Set up Document AI client
        opts = {"api_endpoint": f"{DOCUMENT_AI_LOCATION}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        processor_name = client.processor_path(
            DOCUMENT_AI_PROJECT_ID, 
            DOCUMENT_AI_LOCATION, 
            DOCUMENT_AI_PROCESSOR_ID
        )
        
        # Read the file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Process the document
        raw_document = documentai.RawDocument(content=content, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = client.process_document(request)
        document = result.document
        
        # Extract words and bounding boxes
        text = document.text
        words = []
        bounding_polys = []
        
        for page in document.pages:
            for token in page.tokens:
                bounding_polys.append(token.layout.bounding_poly)
                segs = token.layout.text_anchor.text_segments
                for seg in segs:
                    words.append(text[seg.start_index:seg.end_index])
        
        return document, words, bounding_polys
    
    except Exception as e:
        logger.error(f"Error in document extraction: {str(e)}")
        raise

def generate_and_store_ai_evaluation(submission_id):
    submission = Submission.query.get(submission_id)
    if not submission:
        return None
    
    # Check if evaluation already exists
    existing_evaluation = AIEvaluation.query.filter_by(submission_id=submission_id).first()
    if existing_evaluation:
        return existing_evaluation
    
    # If no text content available, can't evaluate
    if not submission.text_content:
        return None
    
    # Get the assignment details for context
    assignment = Assignment.query.get(submission.assignment_id)
    ideal_answer = None
    if assignment and assignment.ideal_answer_text:
        ideal_answer = assignment.ideal_answer_text
    
    try:
        # Your existing functions that generate HTML output
        if ideal_answer:
            evaluation_html = get_ai_evaluation_with_reference(
                submission.text_content, 
                ideal_answer
            )
        else:
            evaluation_html = get_ai_evaluation(
                submission.text_content
            )
        summary_html = get_ai_summary(submission.text_content)
        
        # Store the evaluation
        evaluation = AIEvaluation(
            submission_id=submission_id,
            summary=summary_html,
            evaluation=evaluation_html
        )
        
        db.session.add(evaluation)
        db.session.commit()
        
        return evaluation
    except Exception as e:
        db.session.rollback()
        print(f"Error generating AI evaluation: {str(e)}")
        return None

class PlagiarismDetector:
    def __init__(self):
        self.punctuation_regex = re.compile(r'[^\w\s]')
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = self.punctuation_regex.sub('', text)
        return ' '.join(text.split())
    
    def get_ngrams(self, text: str, n: int) -> List[str]:
        """Generate n-grams from text."""
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        freq1 = defaultdict(int)
        freq2 = defaultdict(int)
        
        for word in text1.split():
            freq1[word] += 1
        for word in text2.split():
            freq2[word] += 1
            
        common_words = set(freq1.keys()) & set(freq2.keys())
        numerator = sum(freq1[word] * freq2[word] for word in common_words)
        
        mag1 = math.sqrt(sum(freq1[word]**2 for word in freq1))
        mag2 = math.sqrt(sum(freq2[word]**2 for word in freq2))
        
        return numerator / (mag1 * mag2) if mag1 and mag2 else 0
    
    def find_exact_matches(self, text1: str, text2: str, min_length: int = 5) -> List[Tuple[str, int, int]]:
        """Find exact matching phrases between texts."""
        matcher = SequenceMatcher(None, text1.split(), text2.split())
        matches = []
        
        for block in matcher.get_matching_blocks():
            if block.size >= min_length:
                matched_text = ' '.join(text1.split()[block.a:block.a + block.size])
                matches.append((matched_text, block.a, block.b))
                
        return matches
    
    def analyze_similarity(self, text1: str, text2: str) -> Dict:
        """Perform comprehensive similarity analysis between two texts."""
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        sequence_ratio = SequenceMatcher(None, processed_text1, processed_text2).ratio()
        cosine_sim = self.cosine_similarity(processed_text1, processed_text2)
        
        trigrams1 = set(self.get_ngrams(processed_text1, 3))
        trigrams2 = set(self.get_ngrams(processed_text2, 3))
        trigram_overlap = len(trigrams1.intersection(trigrams2)) / max(len(trigrams1), len(trigrams2)) if max(len(trigrams1), len(trigrams2)) > 0 else 0
        
        exact_matches = self.find_exact_matches(processed_text1, processed_text2)
        
        return {
            'sequence_similarity': round(sequence_ratio * 100, 2),
            'cosine_similarity': round(cosine_sim * 100, 2),
            'trigram_overlap': round(trigram_overlap * 100, 2),
            'exact_matches': [(match, pos1, pos2) for match, pos1, pos2 in exact_matches],
            'plagiarism_likelihood': round((sequence_ratio + cosine_sim + trigram_overlap) / 3 * 100, 2),
            'total_matches': len(exact_matches)
        }

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///classroom.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
#all models
class PlagiarismResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    compared_submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    sequence_similarity = db.Column(db.Float)
    cosine_similarity = db.Column(db.Float)
    trigram_overlap = db.Column(db.Float)
    plagiarism_likelihood = db.Column(db.Float)
    total_matches = db.Column(db.Integer)
    exact_matches = db.Column(db.Text)  # Store as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    mail=db.Column(db.String(100), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(10), nullable=False, unique=True)
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
class ClassEnrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    is_teacher = db.Column(db.Boolean, default=False)    
class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    attachment = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  
class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    attachment = db.Column(db.String(200))
    due_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ideal_answer_attachment = db.Column(db.String(200), nullable=True)
    ideal_answer_text = db.Column(db.Text, nullable=True)
    ideal_answer_visible = db.Column(db.Boolean, default=False)
class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    attachment = db.Column(db.String(200))
    comment = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    text_content = db.Column(db.Text, nullable=True)
class AIEvaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    summary = db.Column(db.Text)
    evaluation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    submission = db.relationship('Submission', backref=db.backref('ai_evaluation', uselist=False))
# Routes

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('initial.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(username=username).first()
        
        if existing_user:
            flash('Username already exists. Please choose a different one.')
            return render_template('signup.html')
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        user = User.query.filter_by(username=username).first()
        session['user_id'] = user.id
        session['username'] = user.username
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    enrollments = ClassEnrollment.query.filter_by(user_id=user_id).all()
    
    classes = []
    for enrollment in enrollments:
        class_obj = Class.query.get(enrollment.class_id)
        if class_obj:
            classes.append({
                'id': class_obj.id,
                'title': class_obj.title,
                'is_teacher': enrollment.is_teacher
            })
    
    return render_template('dashboard.html', classes=classes)

@app.route('/create_class', methods=['GET', 'POST'])
def create_class():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        import random, string
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
        new_class = Class(title=title, code=code, creator_id=session['user_id'])
        db.session.add(new_class)
        db.session.flush()
        
        enrollment = ClassEnrollment(user_id=session['user_id'], class_id=new_class.id, is_teacher=True)
        db.session.add(enrollment)
        db.session.commit()
        
        flash('Class created successfully!')
        return redirect(url_for('dashboard'))
    
    return render_template('create_class.html')

@app.route('/join_class', methods=['GET', 'POST'])
def join_class():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        code = request.form.get('code')
        
        class_obj = Class.query.filter_by(code=code).first()
        
        if not class_obj:
            flash('Invalid class code.')
            return render_template('join_class.html')
        
        existing_enrollment = ClassEnrollment.query.filter_by(
            user_id=session['user_id'], class_id=class_obj.id
        ).first()
        
        if existing_enrollment:
            flash('You are already enrolled in this class.')
            return redirect(url_for('dashboard'))
        
        enrollment = ClassEnrollment(user_id=session['user_id'], class_id=class_obj.id, is_teacher=False)
        db.session.add(enrollment)
        db.session.commit()
        
        flash('Joined class successfully!')
        return redirect(url_for('dashboard'))
    
    return render_template('join_class.html')

@app.route('/class/<int:class_id>')
def view_class(class_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    class_obj = Class.query.get_or_404(class_id)
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_id
    ).first_or_404()
    
    announcements = Announcement.query.filter_by(class_id=class_id).order_by(Announcement.created_at.desc()).all()
    assignments = Assignment.query.filter_by(class_id=class_id).order_by(Assignment.created_at.desc()).all()
    
    return render_template('class.html', 
                          class_obj=class_obj, 
                          is_teacher=enrollment.is_teacher,
                          announcements=announcements,
                          assignments=assignments)

@app.route('/class/<int:class_id>/announcement', methods=['GET', 'POST'])
def create_announcement(class_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_id, is_teacher=True
    ).first_or_404()
    
    if request.method == 'POST':
        content = request.form.get('content')
        
        attachment_filename = None
        if 'attachment' in request.files:
            file = request.files['attachment']
            if file.filename:
                attachment_filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], attachment_filename))
        
        announcement = Announcement(
            class_id=class_id,
            user_id=session['user_id'],
            content=content,
            attachment=attachment_filename
        )
        
        db.session.add(announcement)
        db.session.commit()
        
        flash('Announcement created successfully!')
        return redirect(url_for('view_class', class_id=class_id))
    
    return render_template('create_announcement.html', class_id=class_id)

@app.route('/class/<int:class_id>/assignment', methods=['GET', 'POST'])
def create_assignment(class_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_id, is_teacher=True
    ).first_or_404()
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        due_date = request.form.get('due_date')
        ideal_answer_visible = 'ideal_answer_visible' in request.form
        
        attachment_filename = None
        if 'attachment' in request.files:
            file = request.files['attachment']
            if file.filename:
                attachment_filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], attachment_filename))
        
        ideal_answer_filename = None
        ideal_answer_text = None
        if 'ideal_answer' in request.files:
            file = request.files['ideal_answer']
            if file.filename:
                ideal_answer_filename = secure_filename(file.filename)
                ideal_answer_path = os.path.join(app.config['UPLOAD_FOLDER'], ideal_answer_filename)
                file.save(ideal_answer_path)
                
                # If it's a PDF, extract text
                if ideal_answer_path.lower().endswith('.pdf'):
                    try:
                        document, words, _ = extract_document_text(ideal_answer_path)
                        ideal_answer_text = document.text
                    except Exception as e:
                        logger.error(f"Error extracting text from ideal answer: {str(e)}")
        
        due_date_obj = datetime.strptime(due_date, '%Y-%m-%dT%H:%M') if due_date else None
        
        assignment = Assignment(
            class_id=class_id,
            title=title,
            description=description,
            attachment=attachment_filename,
            due_date=due_date_obj,
            ideal_answer_attachment=ideal_answer_filename,
            ideal_answer_text=ideal_answer_text,
            ideal_answer_visible=ideal_answer_visible
        )
        
        db.session.add(assignment)
        db.session.commit()
        
        flash('Assignment created successfully!')
        return redirect(url_for('view_class', class_id=class_id))
    
    return render_template('create_assignment.html', class_id=class_id)
@app.route('/evaluation/<int:submission_id>')
def view_ai_evaluation(submission_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get submission or return 404
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    class_obj = Class.query.get_or_404(assignment.class_id)
    
    # Check if user is authorized (teacher of the class or owner of submission)
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_obj.id
    ).first()
    
    if not enrollment or (not enrollment.is_teacher and session['user_id'] != submission.user_id):
        flash('You are not authorized to view this evaluation.')
        return redirect(url_for('dashboard'))
    
    # Get existing evaluation or generate a new one
    evaluation = AIEvaluation.query.filter_by(submission_id=submission_id).first()
    
    if not evaluation and submission.text_content:
        evaluation = generate_and_store_ai_evaluation(submission_id)
    
    if not evaluation:
        flash('No evaluation available for this submission.')
        return redirect(url_for('view_submission', submission_id=submission_id))
    
    # Get the user who made the submission
    submitter = User.query.get(submission.user_id)
    
    return render_template('ai_review.html',
                          submission=submission,
                          assignment=assignment,
                          class_obj=class_obj,
                          evaluation=evaluation,
                          submitter=submitter,
                          datetime=datetime)

@app.route('/submission/<int:submission_id>')
def view_submission(submission_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    class_obj = Class.query.get_or_404(assignment.class_id)
    
    # Check if user is authorized (teacher of the class or owner of submission)
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_obj.id
    ).first_or_404()
    
    if not enrollment.is_teacher and session['user_id'] != submission.user_id:
        flash('You are not authorized to view this submission.')
        return redirect(url_for('dashboard'))
    
    student = User.query.get(submission.user_id)
    
    return render_template('view_submission.html',
                       submission=submission,
                       assignment=assignment,
                       class_obj=class_obj,
                       student=student)

@app.route('/assignment/<int:assignment_id>')
def view_assignment(assignment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    class_obj = Class.query.get_or_404(assignment.class_id)
    
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_obj.id
    ).first_or_404()
    
    submission = None
    if not enrollment.is_teacher:
        submission = Submission.query.filter_by(
            assignment_id=assignment_id, user_id=session['user_id']
        ).first()
    else:
        # Teacher view logic remains the same
        submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
        users = {sub.user_id: User.query.get(sub.user_id).username for sub in submissions}
        
        # Get plagiarism data for each submission
        plagiarism_data = {}
        for sub in submissions:
            highest_result = PlagiarismResult.query.filter_by(submission_id=sub.id).order_by(
                PlagiarismResult.plagiarism_likelihood.desc()
            ).first()
            
            if highest_result:
                plagiarism_data[sub.id] = {
                    'likelihood': highest_result.plagiarism_likelihood,
                    'compared_with': highest_result.compared_submission_id
                }
        
        return render_template('view_assignment_teacher.html', 
                              assignment=assignment, 
                              class_obj=class_obj,
                              submissions=submissions,
                              users=users,
                              plagiarism_data=plagiarism_data,
                              datetime=datetime)  # Pass datetime module here
    
    return render_template('view_assignment.html', 
                          assignment=assignment, 
                          class_obj=class_obj,
                          submission=submission,
                          datetime=datetime)  # Pass datetime module here

@app.template_filter('from_json')
def from_json(value):
    return json.loads(value) if value else []

@app.route('/assignment/<int:assignment_id>/submit', methods=['GET', 'POST'])
def submit_assignment(assignment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    class_obj = Class.query.get_or_404(assignment.class_id)
    
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_obj.id, is_teacher=False
    ).first_or_404()
    
    existing_submission = Submission.query.filter_by(
        assignment_id=assignment_id, user_id=session['user_id']
    ).first()
    
    if request.method == 'POST':
        comment = request.form.get('comment')
        
        attachment_filename = None
        attachment_path = None
        if 'attachment' in request.files:
            file = request.files['attachment']
            if file.filename:
                attachment_filename = secure_filename(file.filename)
                attachment_path = os.path.join(app.config['UPLOAD_FOLDER'], attachment_filename)
                file.save(attachment_path)
        
        if existing_submission:
            submission = existing_submission
            submission.comment = comment
            if attachment_filename:
                submission.attachment = attachment_filename
                submission.text_content = None  # Mark text for re-extraction
            submission.submitted_at = datetime.utcnow()
        else:
            submission = Submission(
                assignment_id=assignment_id,
                user_id=session['user_id'],
                comment=comment,
                attachment=attachment_filename,
                submitted_at=datetime.utcnow()
            )
            db.session.add(submission)
            db.session.flush()  # Get ID for plagiarism results

        # Process only if there's a PDF and it hasn't been extracted yet
        if attachment_path and attachment_path.lower().endswith('.pdf'):
            try:
                if not submission.text_content:
                    document, words, _ = extract_document_text(attachment_path)
                    submission.text_content = document.text
                
                submitted_text = submission.text_content

                # Get all other submissions for this assignment
                other_submissions = Submission.query.filter(
                    Submission.assignment_id == assignment_id,
                    Submission.id != submission.id,
                    Submission.attachment.isnot(None),
                    Submission.text_content.isnot(None)
                ).all()

                detector = PlagiarismDetector()

                for other_sub in other_submissions:
                    try:
                        other_text = other_sub.text_content
                        results = detector.analyze_similarity(submitted_text, other_text)

                        plagiarism_result = PlagiarismResult(
                            submission_id=submission.id,
                            compared_submission_id=other_sub.id,
                            sequence_similarity=results['sequence_similarity'],
                            cosine_similarity=results['cosine_similarity'],
                            trigram_overlap=results['trigram_overlap'],
                            plagiarism_likelihood=results['plagiarism_likelihood'],
                            total_matches=results['total_matches'],
                            exact_matches=json.dumps(results['exact_matches'])
                        )
                        db.session.add(plagiarism_result)
                    except Exception as e:
                        logger.error(f"Error comparing with submission {other_sub.id}: {str(e)}")

            except Exception as e:
                logger.error(f"Error in plagiarism detection: {str(e)}")
                flash('Assignment submitted, but plagiarism detection encountered an error.')
        
        db.session.commit()
        flash('Assignment submitted successfully!')
        return redirect(url_for('view_assignment', assignment_id=assignment_id))
    
    return render_template('submit_assignment.html', 
                          assignment=assignment, 
                          class_obj=class_obj,
                          existing_submission=existing_submission)

@app.route('/submission/<int:submission_id>/plagiarism')
def view_plagiarism_results(submission_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    class_obj = Class.query.get_or_404(assignment.class_id)
    
    # Check if user is authorized (teacher of the class or owner of submission)
    enrollment = ClassEnrollment.query.filter_by(
        user_id=session['user_id'], class_id=class_obj.id
    ).first_or_404()
    
    if not enrollment.is_teacher and session['user_id'] != submission.user_id:
        flash('You are not authorized to view these results.')
        return redirect(url_for('dashboard'))
    
    # Get plagiarism results
    plagiarism_results = PlagiarismResult.query.filter_by(submission_id=submission_id).all()
    
    # Get usernames for compared submissions
    compared_submissions = {}
    for result in plagiarism_results:
        compared_sub = Submission.query.get(result.compared_submission_id)
        compared_user = User.query.get(compared_sub.user_id)
        compared_submissions[result.compared_submission_id] = {
            'username': compared_user.username,
            'submission_date': compared_sub.submitted_at
        }
    # Add current submission user as well (for "Submission by" line in template)
    submitter = User.query.get(submission.user_id)
    compared_submissions[submission.id] = {
        'username': submitter.username,
        'submission_date': submission.submitted_at
    }

    
    return render_template('plagiarism_results.html',
                          submission=submission,
                          assignment=assignment,
                          class_obj=class_obj,
                          plagiarism_results=plagiarism_results,
                          compared_submissions=compared_submissions,
                          is_teacher=enrollment.is_teacher)

@app.context_processor
def utility_processor():
    def get_user_classes():
        # Only attempt to get classes if user is logged in
        if 'user_id' in session:
            user_id = session['user_id']
            
            # Get all classes where the user is enrolled (either as teacher or student)
            enrollments = ClassEnrollment.query.filter_by(user_id=user_id).all()
            
            # Get the class objects and add is_teacher attribute
            class_data = []
            for enrollment in enrollments:
                class_obj = Class.query.get(enrollment.class_id)
                if class_obj:
                    # Add is_teacher attribute to the class object
                    class_obj.is_teacher = enrollment.is_teacher
                    class_data.append(class_obj)
            
            return class_data
        return []
    
    return {'get_user_classes': get_user_classes}

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get_or_404(user_id)
    
    # Calculate account statistics
    classes_created = Class.query.filter_by(creator_id=user_id).count()
    
    # Classes joined (excluding those created by the user)
    join_enrollments = ClassEnrollment.query.filter_by(user_id=user_id, is_teacher=False).all()
    classes_joined = len(join_enrollments)
    
    # Calculate account age in days
    account_age_days = (datetime.utcnow() - user.created_at).days
    
    return render_template('profile.html', 
                          user=user, 
                          classes_created=classes_created,
                          classes_joined=classes_joined,
                          account_age_days=account_age_days)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get_or_404(user_id)
    
    # Update email
    email = request.form.get('email')
    if email and email != user.mail:
        # Check if email is already in use
        existing_email = User.query.filter_by(mail=email).first()
        if existing_email and existing_email.id != user_id:
            flash('Email already in use by another account.', 'danger')
            return redirect(url_for('profile'))
        user.mail = email
        flash('Email updated successfully.', 'success')
    
    # Update password if provided
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if current_password and new_password and confirm_password:
        # Verify current password
        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('profile'))
        
        # Verify password confirmation
        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('profile'))
        
        # Update password
        user.password = generate_password_hash(new_password)
        flash('Password updated successfully.', 'success')
    
    db.session.commit()
    return redirect(url_for('profile'))

# Delete account route
@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get_or_404(user_id)
    
    # Verify password
    password = request.form.get('password')
    if not check_password_hash(user.password, password):
        flash('Password is incorrect. Account deletion failed.', 'danger')
        return redirect(url_for('profile'))
    
    try:
        # Use a no_autoflush block to prevent premature flushing
        with db.session.no_autoflush:
            # 1. Delete submissions and their related data
            submissions = Submission.query.filter_by(user_id=user_id).all()
            for submission in submissions:
                # First delete AI evaluations for this submission
                AIEvaluation.query.filter_by(submission_id=submission.id).delete()
                
                # Delete plagiarism results for this submission
                PlagiarismResult.query.filter(
                    (PlagiarismResult.submission_id == submission.id) | 
                    (PlagiarismResult.compared_submission_id == submission.id)
                ).delete()
                
                # Delete the submission itself
                db.session.delete(submission)
            
            # 2. Delete announcements
            Announcement.query.filter_by(user_id=user_id).delete()
            
            # 3. Get all classes created by user
            created_classes = Class.query.filter_by(creator_id=user_id).all()
            for class_obj in created_classes:
                # Delete assignments for this class
                assignments = Assignment.query.filter_by(class_id=class_obj.id).all()
                for assignment in assignments:
                    # Delete submissions for this assignment
                    assignment_submissions = Submission.query.filter_by(assignment_id=assignment.id).all()
                    for submission in assignment_submissions:
                        # Delete AI evaluations first
                        AIEvaluation.query.filter_by(submission_id=submission.id).delete()
                        
                        # Delete plagiarism results
                        PlagiarismResult.query.filter(
                            (PlagiarismResult.submission_id == submission.id) | 
                            (PlagiarismResult.compared_submission_id == submission.id)
                        ).delete()
                        db.session.delete(submission)
                    db.session.delete(assignment)
                
                # Delete announcements for this class
                Announcement.query.filter_by(class_id=class_obj.id).delete()
                
                # Delete enrollments for this class
                ClassEnrollment.query.filter_by(class_id=class_obj.id).delete()
                
                # Delete the class
                db.session.delete(class_obj)
            
            # 4. Delete user's enrollments
            ClassEnrollment.query.filter_by(user_id=user_id).delete()
            
            # 5. Finally, delete the user
            db.session.delete(user)
        
        # Commit all changes at once
        db.session.commit()
        
        # Clear session
        session.clear()
        flash('Your account has been successfully deleted.', 'success')
        return redirect(url_for('index'))
    
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred during account deletion: {str(e)}', 'danger')
        return redirect(url_for('profile'))

# Create the database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


