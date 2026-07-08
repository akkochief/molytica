from flask import Blueprint, render_template, request, jsonify, send_file
import os
import pandas as pd
from datetime import datetime
import traceback
import logging
import re
import json
from werkzeug.utils import secure_filename
import io
import base64

xlsx_bp = Blueprint('xlsx', __name__, url_prefix='/excel_to_csv')
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def get_datasets_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def format_size(bytes):
    if bytes == 0:
        return '0 B'
    k = 1024
    sizes = ['B', 'KB', 'MB', 'GB']
    i = 0
    while bytes >= k and i < len(sizes) - 1:
        bytes /= k
        i += 1
    return f"{bytes:.1f} {sizes[i]}"

@xlsx_bp.route('/')
def index():
    return render_template('xlsx.html')

@xlsx_bp.route('/convert', methods=['POST'])
def convert_excel():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({
                'success': False,
                'message': 'Sadece Excel dosyalari (.xlsx, .xls) yuklenebilir'
            })
        
        options = request.form.get('options', '{}')
        try:
            options = json.loads(options)
        except:
            options = {}
        
        delimiter = options.get('delimiter', ',')
        encoding = options.get('encoding', 'utf-8')
        line_ending = options.get('lineEnding', '\r\n')
        include_header = options.get('includeHeader', True)
        quote_all = options.get('quoteAll', False)
        date_format = options.get('dateFormat', 'yyyy-mm-dd')
        sheet_name = options.get('sheetName', 'auto')
        
        if sheet_name == 'auto':
            df = pd.read_excel(file)
        else:
            df = pd.read_excel(file, sheet_name=sheet_name)
        
        if date_format != 'original':
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if date_format == 'iso':
                        df[col] = df[col].dt.isoformat()
                    else:
                        df[col] = df[col].dt.strftime(date_format.replace('yyyy', '%Y').replace('mm', '%m').replace('dd', '%d'))
        
        csv_buffer = io.StringIO()
        df.to_csv(
            csv_buffer,
            sep=delimiter,
            index=False,
            header=include_header,
            quoting=1 if quote_all else 0,
            line_terminator=line_ending
        )
        
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        base_name = os.path.splitext(file.filename)[0]
        csv_filename = f"{base_name}.csv"
        
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        filepath = os.path.join(output_dir, csv_filename)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(csv_content)
        
        return jsonify({
            'success': True,
            'message': 'Dosya basariyla donusturuldu',
            'csv_content': csv_content,
            'csv_filename': csv_filename,
            'filepath': filepath,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'preview': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Donusturme hatasi: {str(e)}'
        })

@xlsx_bp.route('/save', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        filename = data.get('filename', 'donusturulen_veri')
        content = data.get('content', '')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Kaydedilecek veri yok'
            })
        
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        filename = sanitize_filename(filename)
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({
            'success': True,
            'message': f'Dosya basariyla kaydedildi: {filename}',
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Kaydetme hatasi: {str(e)}'
        })

@xlsx_bp.route('/list', methods=['GET'])
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
                'size_str': format_size(stat.st_size),
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

@xlsx_bp.route('/delete', methods=['POST'])
def delete_csv():
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

@xlsx_bp.route('/download/<filename>', methods=['GET'])
def download_csv(filename):
    try:
        filename = sanitize_filename(filename)
        filepath = os.path.join('static/datasets', filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            }), 404
        
        return send_file(
            filepath, 
            as_attachment=True, 
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@xlsx_bp.route('/preview', methods=['POST'])
def preview_csv():
    try:
        data = request.get_json()
        content = data.get('content', '')
        delimiter = data.get('delimiter', ',')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Onizlenecek veri yok'
            })
        
        from io import StringIO
        df = pd.read_csv(StringIO(content), sep=delimiter)
        
        return jsonify({
            'success': True,
            'headers': df.columns.tolist(),
            'preview': df.head(10).to_dict('records'),
            'row_count': len(df),
            'column_count': len(df.columns)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@xlsx_bp.route('/validate', methods=['POST'])
def validate_csv():
    try:
        data = request.get_json()
        content = data.get('content', '')
        delimiter = data.get('delimiter', ',')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Dogrulanacak veri yok'
            })
        
        from io import StringIO
        df = pd.read_csv(StringIO(content), sep=delimiter)
        
        errors = []
        warnings = []
        
        for col in df.columns:
            if df[col].isnull().all():
                warnings.append(f"'{col}' sutunu tamamen bos")
        
        numeric_indicators = ['amount', 'yield', 'temp', 'time', 'minute', 'cycle', 'centigrades', 'quantity']
        for col in df.columns:
            col_lower = col.lower()
            if any(ind in col_lower for ind in numeric_indicators):
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    warnings.append(f"'{col}' sutunu sayisal degerler icermiyor")
        
        for col in df.columns:
            if 'yield' in col.lower():
                try:
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    out_of_range = numeric[(numeric < 0) | (numeric > 100)]
                    if not out_of_range.empty:
                        warnings.append(f"'{col}' sutununda {len(out_of_range)} satir 0-100 araligi disinda")
                except:
                    pass
        
        return jsonify({
            'success': True,
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'row_count': len(df),
            'column_count': len(df.columns)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@xlsx_bp.route('/export', methods=['POST'])
def export_csv():
    try:
        data = request.get_json()
        content = data.get('content', '')
        filename = data.get('filename', 'donusturulen_veri')
        
        if not content:
            return jsonify({
                'success': False,
                'message': 'Dis aktarilacak veri yok'
            })
        
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        filename = sanitize_filename(filename)
        
        temp_dir = 'static/temp'
        ensure_directory(temp_dir)
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Dis aktarma hatasi: {str(e)}'
        }), 500

@xlsx_bp.route('/sheets', methods=['POST'])
def get_sheets():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        
        return jsonify({
            'success': True,
            'sheets': sheet_names
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@xlsx_bp.route('/info', methods=['POST'])
def get_excel_info():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        excel_file = pd.ExcelFile(file)
        
        info = {
            'filename': file.filename,
            'size': format_size(file_size),
            'sheets': excel_file.sheet_names,
            'sheet_count': len(excel_file.sheet_names)
        }
        
        file.seek(0)
        df = pd.read_excel(file, sheet_name=0, nrows=1)
        info['first_sheet_columns'] = df.columns.tolist()
        info['first_sheet_column_count'] = len(df.columns)
        
        return jsonify({
            'success': True,
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@xlsx_bp.route('/rename', methods=['POST'])
def rename_csv():
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

@xlsx_bp.route('/stats', methods=['GET'])
def get_stats():
    try:
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        
        files = get_datasets_from_folder(output_dir)
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_files': len(files),
                'total_size': total_size,
                'total_size_str': format_size(total_size),
                'last_modified': datetime.fromtimestamp(max(os.path.getmtime(os.path.join(output_dir, f)) for f in files)).strftime('%Y-%m-%d %H:%M') if files else 'Henuz dosya yok'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@xlsx_bp.route('/convert_advanced', methods=['POST'])
def convert_advanced():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Dosya secilmedi'
            })
        
        data = request.form.get('data', '{}')
        try:
            options = json.loads(data)
        except:
            options = {}
        
        sheet_name = options.get('sheet_name', 0)
        skip_rows = options.get('skip_rows', 0)
        use_cols = options.get('use_cols', None)
        
        if use_cols:
            use_cols = [int(c) if c.isdigit() else c for c in use_cols.split(',')] if isinstance(use_cols, str) else use_cols
        
        df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            skiprows=skip_rows,
            usecols=use_cols,
            engine='openpyxl'
        )
        
        delimiter = options.get('delimiter', ',')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, sep=delimiter, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        base_name = os.path.splitext(file.filename)[0]
        csv_filename = f"{base_name}_advanced.csv"
        
        output_dir = 'static/datasets'
        ensure_directory(output_dir)
        filepath = os.path.join(output_dir, csv_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        return jsonify({
            'success': True,
            'message': 'Dosya basariyla donusturuldu',
            'csv_content': csv_content,
            'csv_filename': csv_filename,
            'filepath': filepath,
            'row_count': len(df),
            'column_count': len(df.columns)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Donusturme hatasi: {str(e)}'
        })