from flask import Blueprint, render_template, request, jsonify, send_file
import os
import pandas as pd
from datetime import datetime
import traceback
import logging
import re
from werkzeug.utils import secure_filename

csv_bp = Blueprint('csv', __name__, url_prefix='/upload-csv')
logger = logging.getLogger(__name__)

def sanitize_csv_filename(filename):
    filename = os.path.basename(filename)
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def get_datasets_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

@csv_bp.route('/')
def index():
    return render_template('csv.html')

@csv_bp.route('/save', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        filename = data.get('filename', 'edited_data')
        content = data.get('data', '')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'No data to save'
            })
        
        filename = sanitize_csv_filename(filename)
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({
            'success': True,
            'message': f'File saved successfully: {filename}',
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Save error: {str(e)}'
        })

@csv_bp.route('/upload', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file selected'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            })
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                'success': False,
                'message': 'Only CSV files can be uploaded'
            })
        
        content = file.read().decode('utf-8')
        
        lines = content.strip().split('\n')
        if not lines:
            return jsonify({
                'success': False,
                'message': 'File is empty'
            })
        
        headers = [h.strip() for h in lines[0].split(',')]
        rows = []
        for line in lines[1:]:
            if line.strip():
                values = [v.strip() for v in line.split(',')]
                row = {}
                for i, header in enumerate(headers):
                    row[header] = values[i] if i < len(values) else ''
                rows.append(row)
        
        return jsonify({
            'success': True,
            'headers': headers,
            'rows': rows,
            'filename': file.filename
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Upload error: {str(e)}'
        })

@csv_bp.route('/load/<filename>', methods=['GET'])
def load_csv(filename):
    try:
        filename = sanitize_csv_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'File not found: {filename}'
            })
        
        df = pd.read_csv(filepath)
        df = df.fillna('')
        
        return jsonify({
            'success': True,
            'headers': df.columns.tolist(),
            'rows': df.to_dict('records'),
            'filename': filename
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Load error: {str(e)}'
        })

@csv_bp.route('/list', methods=['GET'])
def list_csv_files():
    try:
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        files = []
        for filename in get_datasets_from_folder(output_dir):
            filepath = os.path.join(output_dir, filename)
            stat = os.stat(filepath)
            files.append({
                'name': filename,
                'size': stat.st_size,
                'size_str': f"{stat.st_size // 1024} KB" if stat.st_size > 1024 else f"{stat.st_size} B",
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@csv_bp.route('/delete', methods=['POST'])
def delete_csv():
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({
                'success': False,
                'message': 'Filename is required'
            })
        
        filename = sanitize_csv_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'File not found: {filename}'
            })
        
        os.remove(filepath)
        return jsonify({
            'success': True,
            'message': f'{filename} deleted'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@csv_bp.route('/preview', methods=['POST'])
def preview_csv():
    try:
        data = request.get_json()
        headers = data.get('headers', [])
        rows = data.get('rows', [])
        
        if not headers or not rows:
            return jsonify({
                'success': False,
                'message': 'No data to preview'
            })
        
        preview_rows = rows[:10]
        
        return jsonify({
            'success': True,
            'headers': headers,
            'preview': preview_rows,
            'total': len(rows),
            'preview_count': len(preview_rows)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@csv_bp.route('/validate', methods=['POST'])
def validate_csv():
    try:
        data = request.get_json()
        headers = data.get('headers', [])
        rows = data.get('rows', [])
        
        if not headers:
            return jsonify({
                'success': False,
                'message': 'No data to validate'
            })
        
        errors = []
        warnings = []
        
        for i, header in enumerate(headers):
            if not header or header.strip() == '':
                errors.append(f"Column {i+1}: Column name is empty")
        
        numeric_indicators = ['amount', 'yield', 'temp', 'time', 'minute', 'cycle', 'centigrades', 'quantity']
        
        for i, row in enumerate(rows):
            for header in headers:
                if header not in row:
                    errors.append(f"Row {i+1}: '{header}' column is missing")
            
            for header in headers:
                header_lower = header.lower()
                if any(ind in header_lower for ind in numeric_indicators):
                    if header in row and row[header]:
                        try:
                            float(row[header])
                        except (ValueError, TypeError):
                            warnings.append(f"Row {i+1}: '{header}' value should be numeric (value: {row[header]})")
        
        return jsonify({
            'success': True,
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(rows),
            'column_count': len(headers)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@csv_bp.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    try:
        filename = sanitize_csv_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'File not found: {filename}'
            }), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@csv_bp.route('/export', methods=['POST'])
def export_csv():
    try:
        data = request.get_json()
        headers = data.get('headers', [])
        rows = data.get('rows', [])
        filename = data.get('filename', 'data')
        
        if not headers or not rows:
            return jsonify({
                'success': False,
                'message': 'No data to export'
            })
        
        csv_lines = [','.join(headers)]
        for row in rows:
            values = []
            for header in headers:
                value = row.get(header, '')
                if isinstance(value, str) and (',' in value or '"' in value or '\n' in value):
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                values.append(str(value))
            csv_lines.append(','.join(values))
        
        csv_content = '\n'.join(csv_lines)
        filename = sanitize_csv_filename(filename)
        
        temp_dir = 'static/temp'
        ensure_directory(temp_dir)
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Export error: {str(e)}'
        }), 500

@csv_bp.route('/rename', methods=['POST'])
def rename_csv():
    try:
        data = request.get_json()
        old_name = data.get('old_name', '')
        new_name = data.get('new_name', '')
        
        if not old_name or not new_name:
            return jsonify({
                'success': False,
                'message': 'Old and new filenames are required'
            })
        
        old_name = sanitize_csv_filename(old_name)
        new_name = sanitize_csv_filename(new_name)
        
        old_path = os.path.join('static/datasets', old_name)
        new_path = os.path.join('static/datasets', new_name)
        
        if not os.path.exists(old_path):
            return jsonify({
                'success': False,
                'message': f'File not found: {old_name}'
            })
        
        if os.path.exists(new_path):
            return jsonify({
                'success': False,
                'message': f'{new_name} already exists'
            })
        
        os.rename(old_path, new_path)
        
        return jsonify({
            'success': True,
            'message': f'{old_name} renamed to {new_name}',
            'new_name': new_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@csv_bp.route('/info/<filename>', methods=['GET'])
def csv_info(filename):
    try:
        filename = sanitize_csv_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'File not found: {filename}'
            })
        
        df = pd.read_csv(filepath)
        
        info = {
            'filename': filename,
            'size': os.path.getsize(filepath),
            'size_str': f"{os.path.getsize(filepath) // 1024} KB" if os.path.getsize(filepath) > 1024 else f"{os.path.getsize(filepath)} B",
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M')
        }
        
        return jsonify({
            'success': True,
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })