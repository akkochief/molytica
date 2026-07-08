import os
import logging
import subprocess
import threading
import glob
from datetime import datetime
import pandas as pd
import re
import requests
from urllib.parse import urlparse
import platform
import ctypes
import locale
import tempfile
from werkzeug.utils import secure_filename
import json
import traceback
import warnings
import xml.etree.ElementTree as ET
from flask import Flask, render_template, request, jsonify, Response, send_file

from routes.dataset_routes import dataset_bp
from routes.manual_routes import manual_bp
from routes.csv_routes import csv_bp
from routes.xlsx_routes import xlsx_bp
from routes.predict_ml_routes import predict_ml_bp
from routes.predict_routes import predict_bp
from routes.visual_routes import visual_bp
from routes.compare_routes import compare_bp
from routes.help_routes import help_bp

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
logging.basicConfig(level=logging.DEBUG)

app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['OUTPUTS_DIR'] = 'static/datasets'
app.config['TRAINED_MODELS_DIR'] = 'static/models'
app.config['DATASET_LOG_DIR2'] = 'static/datasets'
app.config['CODE_LOG_DIR'] = 'code'
app.config['IMAGES_LOG_DIR'] = 'static/images'
app.config['RESULTS_DIR'] = 'static/outputs'
app.config['SAVE_DIR'] = 'static/datasets'

app.register_blueprint(dataset_bp)
app.register_blueprint(manual_bp)
app.register_blueprint(csv_bp)
app.register_blueprint(xlsx_bp)
app.register_blueprint(predict_ml_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(visual_bp)
app.register_blueprint(compare_bp)
app.register_blueprint(help_bp)

rowcount = 0
progress_data = {}
smiles_cache = {}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)
os.makedirs(app.config['TRAINED_MODELS_DIR'], exist_ok=True)
os.makedirs(app.config['IMAGES_LOG_DIR'], exist_ok=True)
os.makedirs(app.config['CODE_LOG_DIR'], exist_ok=True)
os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)
os.makedirs('xml_data', exist_ok=True)
os.makedirs('static/model', exist_ok=True)
os.makedirs('static/temp', exist_ok=True)
os.makedirs('static/xml_data', exist_ok=True)
os.makedirs('static/feedback', exist_ok=True)

STD_OUTPUT_HANDLE = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

class WindowsConsole:
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.setup_console()
    
    def setup_console(self):
        handle = self.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        self.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
        self.kernel32.SetConsoleMode(handle, mode)

if platform.system() == "Windows":
    windows_console = WindowsConsole()
    locale.setlocale(locale.LC_ALL, 'turkish')

def get_datasets_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def get_codes_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.py')]

def merge_csv_to_output(input_files, output_file='out.csv'):
    global rowcount
    try:
        dfs = [pd.read_csv(f"static/datasets/{f}") for f in input_files]
        merged_df = pd.concat(dfs, ignore_index=True)
        rowcount = len(merged_df)
        merged_df.to_csv(output_file, index=False)
        merged_df.to_csv('static/model/miaw.csv', index=False)
        return rowcount
    except Exception as e:
        print(f"Merge error: {str(e)}")
        return 0

def get_rowcount_for_datasets(dataset_files):
    try:
        total_rows = 0
        for dataset_file in dataset_files:
            file_path = os.path.join("static/datasets", dataset_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                total_rows += len(df)
        return total_rows
    except Exception as e:
        print(f"Row count error: {str(e)}")
        return 0

def run_python_script(script_name, datasets):
    command = ['python', f'code/{script_name}'] + datasets
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        app.logger.info(line.strip())
    process.wait()

def sanitize_csv_filename(filename):
    filename = os.path.basename(filename)
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    return re.sub(r'[^\w\-_.]', '_', filename)

def standardize_smiles(smiles):
    return smiles

def get_pubchem_smiles(name):
    return name

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dataset')
def analyze_dataset():
    return render_template('dataset.html')

@app.route('/edit-xml')
def edit_xml():
    return render_template('edit_var.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict_ml')
def predict_ml():
    return render_template('predict_ml.html')

@app.route('/visual')
def visual():
    return render_template('visual.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/edit_dataset')
def edit_dataset():
    return render_template('edit.html')

@app.route('/preview_dataset')
def preview_dataset():
    return render_template('preview.html')

@app.route('/excel_to_csv')
def excel_to_csv():
    return render_template('xlsx.html')

@app.route('/manual')
def manuel():
    return render_template('manual.html')

@app.route('/upload-csv')
def upload_csv():
    return render_template('csv.html')

@app.route('/compare-models')
def compare_models():
    return render_template('compare.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        if file and file.filename.endswith('.csv'):
            filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['OUTPUTS_DIR'], filename)
            os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)
            file.save(filepath)
            return jsonify({'success': True, 'message': f'File uploaded: {filename}'})
        return jsonify({'success': False, 'message': 'Only CSV files'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/progress/<task_id>')
def get_progress(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(progress_data[task_id])

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        safe_path = os.path.join(app.root_path, filename)
        if not os.path.exists(safe_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(safe_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    df.to_csv(output_path, index=False)
    
    return jsonify({
        'success': True,
        'download_url': f'/download/{output_filename}',
        'preview': df.head(10).to_html(classes='table table-striped')
    })

@app.route('/get_dataset_rowcount', methods=['POST'])
def get_dataset_rowcount():
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    if not selected_datasets:
        return jsonify({"rowcount": 0})
    return jsonify({"rowcount": get_rowcount_for_datasets(selected_datasets)})

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    selected_code = data.get('selectedCode', '')
    global rowcount
    rowcount = merge_csv_to_output(selected_datasets, "out.csv")
    if not selected_code:
        return jsonify({"status": "error", "message": "No model script selected."}), 400
    thread = threading.Thread(target=run_python_script, args=(selected_code, selected_datasets))
    thread.start()
    return jsonify({"status": "success", "message": "Model is running!"})

@app.route('/stream_output')
def stream_output():
    selected_code = request.args.get('code')
    def generate():
        if not selected_code:
            yield "data: ERROR: No script selected\n\n"
            return
        command = ['python', f'code/{selected_code}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        while True:
            output = process.stdout.readline()
            if output:
                yield f"data: {output.strip()}\n\n"
            error = process.stderr.readline()
            if error:
                yield f"data: ERROR: {error.strip()}\n\n"
            if process.poll() is not None:
                for out in process.stdout:
                    if out:
                        yield f"data: {out.strip()}\n\n"
                yield "data: COMPLETE\n\n"
                break
    return Response(generate(), content_type='text/event-stream')

@app.route('/get_latest_result')
def get_latest_result():
    try:
        os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)
        result_files = glob.glob(os.path.join(app.config['RESULTS_DIR'], '*.txt'))
        if not result_files:
            return jsonify({"status": "error", "message": "No results found"}), 404
        latest_file = max(result_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            content = f.read()
        return jsonify({"status": "success", "content": content, "filename": os.path.basename(latest_file)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download_model')
def download_model():
    model_path = os.path.join(app.config['TRAINED_MODELS_DIR'], 'models.pkl')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return jsonify({"status": "error", "message": "Model file not found"}), 404

@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        filename = data.get('filename', 'data')
        content = data.get('data', '')
        os.makedirs('static/datasets', exist_ok=True)
        filepath = os.path.join('static/datasets', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'message': 'File saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_xml_files')
def get_xml_files():
    xml_files = []
    try:
        xml_dir = os.path.join(app.root_path, 'xml_data')
        os.makedirs(xml_dir, exist_ok=True)
        for file in glob.glob(os.path.join(xml_dir, '*.xml')):
            xml_files.append(os.path.basename(file))
    except Exception as e:
        app.logger.error(f"XML list error: {str(e)}")
    return jsonify(xml_files)

@app.route('/get_xml_data')
def get_xml_data():
    filename = request.args.get('file', '')
    if not filename:
        return jsonify({"error": "No file"}), 400
    try:
        filepath = os.path.join('xml_data', filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/xml'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-xml', methods=['POST'])
def save_xml():
    try:
        data = request.get_json()
        xml_content = data.get('xml', '')
        try:
            ET.fromstring(xml_content)
        except ET.ParseError as e:
            return jsonify({'success': False, 'message': f'Invalid XML: {str(e)}'})
        xml_path = os.path.join('xml_data', 'info.xml')
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        return jsonify({'success': True, 'message': 'XML saved'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/load-csv', methods=['POST'])
def api_load_csv():
    data = request.get_json()
    url = data.get('url', '')
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'}), 400
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify({'success': True, 'content': response.text, 'filename': os.path.basename(urlparse(url).path)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/execute', methods=['POST'])
def api_execute():
    data = request.json
    command = data.get('command', '').strip()
    if not command:
        return jsonify({'success': False, 'output': 'No command'})
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return jsonify({'success': False, 'output': stderr if stderr else "Command failed"})
        return jsonify({'success': True, 'output': stdout, 'cwd': os.getcwd()})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

@app.route('/api/getcwd', methods=['GET'])
def api_getcwd():
    return jsonify({'cwd': os.getcwd()})

@app.route('/api/get_csv_files')
def api_get_csv_files():
    try:
        output_dir = app.config['OUTPUTS_DIR']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return jsonify({'success': True, 'files': []})
        files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(output_dir, filename)
                size = os.path.getsize(filepath)
                size_str = f"{size // 1024} KB" if size > 1024 else f"{size} B"
                files.append({'name': filename, 'size': size_str})
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)