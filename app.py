from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import requests
import time
import fitz  # PyMuPDF
from spellchecker import SpellChecker
import string
import language_tool_python  # For grammar checking

app = Flask(__name__)
app.template_folder=os.path.abspath('template')



# Configurations
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# Replace with your Azure Computer Vision API key and endpoint


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Use the environment variables
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")



# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def extract_text_from_pdf(pdf_file_path):
    # Your existing extract_text_from_pdf function here...
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }
    
    params = {
        'language': 'en'
    }
    
    with open(pdf_file_path, 'rb') as f:
        data = f.read()
    
    response = requests.post(endpoint, headers=headers, params=params, data=data)
    
    if response.status_code != 202:
        print(f"Error: {response.status_code}, {response.json()}")
        return None
    
    operation_location = response.headers['Operation-Location']
    
    while True:
        result = requests.get(operation_location, headers={'Ocp-Apim-Subscription-Key': subscription_key})
        result_data = result.json()
        
        if result.status_code == 200 and result_data['status'] == 'succeeded':
            break
        elif result_data['status'] == 'failed':
            print("Failed to extract text.")
            return None
        
        print("Waiting for result...")
        time.sleep(2)

    # Store all text and word data
    full_text = ""
    word_data = []
    
    for read_result in result_data['analyzeResult']['readResults']:
        page_number = read_result['page'] - 1
        dpi = 72
        width_in_points = read_result['width'] * dpi
        height_in_points = read_result['height'] * dpi
        
        for line in read_result['lines']:
            full_text += line['text'] + " "
            
            for word in line['words']:
                word_data.append({
                    'text': word['text'],
                    'bbox': word['boundingBox'],
                    'page': page_number,
                    'width': width_in_points,
                    'height': height_in_points,
                    'confidence': word['confidence']
                })
    
    return full_text.strip(), word_data
    pass
def convert_coordinates(bbox, width, height):
    """Convert coordinates from inches to PDF points."""
    dpi = 72
    x1 = bbox[0] * dpi
    y1 = bbox[1] * dpi
    x2 = bbox[2] * dpi
    y2 = bbox[5] * dpi
    return [x1, y1, x2, y2]



# ... (keep other utility functions as they are) ...
def highlight_text(pdf_file_path, results, spell_color, grammar_color, keyword_color):
    pdf_document = fitz.open(pdf_file_path)
    
    # Convert hex colors to RGB tuples
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    # Define colors for different types of highlights
    colors = {
        'spelling': hex_to_rgb(spell_color),
        'grammar': hex_to_rgb(grammar_color),
        'keywords': hex_to_rgb(keyword_color)
    }
    
    # Organize results by page
    highlights_by_page = {}
    for highlight_type, items in results.items():
        for item in items:
            page_num = item['page']
            if page_num not in highlights_by_page:
                highlights_by_page[page_num] = {'spelling': [], 'grammar': [], 'keywords': []}
            highlights_by_page[page_num][highlight_type].append(item)
    
    # Process each page
    for page_num, page_highlights in highlights_by_page.items():
        page = pdf_document[page_num]
        
        # Process each type of highlight
        for highlight_type, items in page_highlights.items():
            color = colors[highlight_type]
            
            for item in items:
                # Convert coordinates to PDF space
                bbox = convert_coordinates(
                    item['bbox'],
                    item['width'],
                    item['height']
                )
                
                # Create rectangle for highlight
                rect = fitz.Rect(bbox)
                
                # Add highlight with appropriate color
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], "highlighted_output.pdf")
    pdf_document.save(output_path)
    pdf_document.close()
    return output_path


def check_text(full_text, word_data, keywords, spell_check=True, grammar_check=True):
    spell = SpellChecker()
    tool = language_tool_python.LanguageTool('en-US')
    
    results = {
        'spelling': [],  # Yellow highlights
        'grammar': [],   # Pink highlights
        'keywords': []   # Green highlights
    }
    
    if grammar_check:
        # Check grammar on the full text
        print("Checking grammar...")
        grammar_matches = tool.check(full_text)
    
    # Process spelling and keywords for each word
    for word_info in word_data:
        word = word_info['text']
        word_clean = word.strip(string.punctuation).lower()
        
        # Check spelling
        if spell_check and word_clean and spell.unknown([word_clean]):
            results['spelling'].append(word_info)
        
        # Check keywords
        for keyword in keywords:
            if keyword.lower() in word_clean:
                results['keywords'].append(word_info)
    
    if grammar_check:
        # Process grammar errors
        for error in grammar_matches:
            error_start = error.offset
            error_length = error.errorLength
            
            # Find the words that correspond to this error
            current_pos = 0
            for word_info in word_data:
                word_text = word_info['text'] + " "
                if current_pos <= error_start < current_pos + len(word_text):
                    error_info = word_info.copy()
                    error_info['error_message'] = error.message
                    results['grammar'].append(error_info)
                    break
                current_pos += len(word_text)
    
    return results

# ... (keep other utility functions as they are) ...

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    spell_check = request.form.get('spell_check') == 'on'
    grammar_check = request.form.get('grammar_check') == 'on'
    keywords = request.form.get('keywords', '').split(',') if request.form.get('keywords') else []
    
    # Get color values
    spell_color = request.form.get('spell_color', '#FFFF00')  # Default to yellow
    grammar_color = request.form.get('grammar_color', '#FFC0CB')  # Default to pink
    keyword_color = request.form.get('keyword_color', '#90EE90')  # Default to light green

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
            
            # Process the PDF
            text_data = extract_text_from_pdf(pdf_path)
            if not text_data:
                return jsonify({'error': 'Failed to extract text from PDF'}), 500

            full_text, word_data = text_data
            results = check_text(full_text, word_data, keywords, spell_check, grammar_check)
            
            output_pdf_path = highlight_text(pdf_path, results, spell_color, grammar_color, keyword_color)
            
            if os.path.exists(output_pdf_path):
                return send_file(output_pdf_path, as_attachment=True, download_name='analyzed_' + filename)
            else:
                return jsonify({'error': 'Failed to generate the PDF'}), 500

        except Exception as e:
            app.logger.error(f"Error processing PDF: {str(e)}")
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

        finally:
            # Clean up temporary files
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                if 'output_pdf_path' in locals() and os.path.exists(output_pdf_path):
                    os.remove(output_pdf_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up files: {str(e)}")

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run(debug=True)