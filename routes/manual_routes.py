from flask import Blueprint, render_template, request, jsonify
import os
import pandas as pd
from datetime import datetime
import traceback
import logging
import re

manual_bp = Blueprint('manual', __name__, url_prefix='/manual')
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    filename = os.path.basename(filename)
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

@manual_bp.route('/')
def index():
    return render_template('manual.html')

@manual_bp.route('/save', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        filename = data.get('filename', 'manual_data')
        csv_content = data.get('data', '')
        
        if not csv_content:
            return jsonify({
                'success': False, 
                'message': 'Kaydedilecek veri yok'
            })
        
        filename = sanitize_filename(filename)
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        return jsonify({
            'success': True,
            'message': f'Veriler basariyla kaydedildi: {filename}',
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Kaydetme hatasi: {str(e)}'
        })

@manual_bp.route('/load', methods=['POST'])
def load_data():
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({
                'success': False,
                'message': 'Dosya adi gerekli'
            })
        
        filepath = os.path.join('static/datasets', filename)
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        df = pd.read_csv(filepath)
        
        return jsonify({
            'success': True,
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'filename': filename
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Yukleme hatasi: {str(e)}'
        })

@manual_bp.route('/list', methods=['GET'])
def list_files():
    try:
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        files = []
        for f in os.listdir(output_dir):
            if f.endswith('.csv'):
                filepath = os.path.join(output_dir, f)
                stat = os.stat(filepath)
                files.append({
                    'name': f,
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

@manual_bp.route('/delete', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({
                'success': False,
                'message': 'Dosya adi gerekli'
            })
        
        filepath = os.path.join('static/datasets', filename)
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        os.remove(filepath)
        return jsonify({
            'success': True,
            'message': f'{filename} silindi'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@manual_bp.route('/preview', methods=['POST'])
def preview_data():
    try:
        data = request.get_json()
        rows = data.get('rows', [])
        
        if not rows:
            return jsonify({
                'success': False,
                'message': 'Onizlenecek veri yok'
            })
        
        return jsonify({
            'success': True,
            'preview': rows[:10],
            'total': len(rows)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@manual_bp.route('/validate', methods=['POST'])
def validate_data():
    try:
        data = request.get_json()
        rows = data.get('rows', [])
        
        if not rows:
            return jsonify({
                'success': False,
                'message': 'Dogrulanacak veri yok'
            })
        
        expected_columns = ['ArB', 'ArX', 'Product', 'Catalizor', 'Base', 
                           'Solv1', 'Solv2', 'Amount', 'Centigrades', 
                           'Minute', 'Cycle', 'Yield']
        
        errors = []
        warnings = []
        
        for i, row in enumerate(rows):
            for col in expected_columns:
                if col not in row:
                    errors.append(f"Satir {i+1}: '{col}' sutunu eksik")
            
            numeric_fields = ['Amount', 'Centigrades', 'Minute', 'Cycle', 'Yield']
            for field in numeric_fields:
                if field in row and row[field]:
                    try:
                        float(row[field])
                    except ValueError:
                        errors.append(f"Satir {i+1}: '{field}' sayisal olmali")
            
            if 'Yield' in row and row['Yield']:
                try:
                    y = float(row['Yield'])
                    if y < 0 or y > 100:
                        warnings.append(f"Satir {i+1}: Yield {y} degeri 0-100 araliginda olmali")
                except:
                    pass
        
        return jsonify({
            'success': True,
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(rows)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@manual_bp.route('/rename', methods=['POST'])
def rename_file():
    try:
        data = request.get_json()
        old_name = data.get('old_name', '')
        new_name = data.get('new_name', '')
        
        if not old_name or not new_name:
            return jsonify({
                'success': False,
                'message': 'Eski ve yeni dosya adi gerekli'
            })
        
        old_name = sanitize_filename(old_name)
        new_name = sanitize_filename(new_name)
        
        old_path = os.path.join('static/datasets', old_name)
        new_path = os.path.join('static/datasets', new_name)
        
        if not os.path.exists(old_path):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {old_name}'
            })
        
        if os.path.exists(new_path):
            return jsonify({
                'success': False,
                'message': f'{new_name} zaten mevcut'
            })
        
        os.rename(old_path, new_path)
        
        return jsonify({
            'success': True,
            'message': f'{old_name} -> {new_name} olarak yenilendi',
            'new_name': new_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@manual_bp.route('/export', methods=['POST'])
def export_data():
    try:
        data = request.get_json()
        rows = data.get('rows', [])
        filename = data.get('filename', 'manual_export')
        
        if not rows:
            return jsonify({
                'success': False,
                'message': 'Dis aktarilacak veri yok'
            })
        
        columns = list(rows[0].keys()) if rows else []
        
        csv_lines = [','.join(columns)]
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, '')
                if isinstance(value, str) and (',' in value or '"' in value or '\n' in value):
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                values.append(str(value))
            csv_lines.append(','.join(values))
        
        csv_content = '\n'.join(csv_lines)
        filename = sanitize_filename(filename)
        
        temp_dir = 'static/temp'
        ensure_directory(temp_dir)
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        from flask import send_file
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Dis aktarma hatasi: {str(e)}'
        }), 500

@manual_bp.route('/info/<filename>', methods=['GET'])
def file_info(filename):
    try:
        filename = sanitize_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        df = pd.read_csv(filepath)
        
        info = {
            'filename': filename,
            'size': os.path.getsize(filepath),
            'size_str': f"{os.path.getsize(filepath) // 1024} KB" if os.path.getsize(filepath) > 1024 else f"{os.path.getsize(filepath)} B",
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
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

@manual_bp.route('/stats', methods=['GET'])
def get_stats():
    try:
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_files': len(files),
                'total_size': total_size,
                'total_size_str': f"{total_size // 1024} KB" if total_size > 1024 else f"{total_size} B",
                'last_modified': datetime.fromtimestamp(max(os.path.getmtime(os.path.join(output_dir, f)) for f in files)).strftime('%Y-%m-%d %H:%M') if files else 'Henüz dosya yok'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })