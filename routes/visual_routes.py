from flask import Blueprint, render_template, request, jsonify, send_file, current_app
import os
import glob
from datetime import datetime
import traceback
import logging
import json
from werkzeug.utils import secure_filename

visual_bp = Blueprint('visual', __name__, url_prefix='/visual')
logger = logging.getLogger(__name__)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def get_image_folders(base_path):
    if not os.path.exists(base_path):
        return []
    
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    return folders

def get_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
    images = []
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()
        if os.path.isfile(file_path) and ext in image_extensions:
            stat = os.stat(file_path)
            images.append({
                'name': file,
                'path': file_path,
                'size': stat.st_size,
                'size_str': format_size(stat.st_size),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    images.sort(key=lambda x: x['modified'], reverse=True)
    return images

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

@visual_bp.route('/')
def index():
    images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
    folders = get_image_folders(images_dir)
    return render_template('visual.html', images=folders)

@visual_bp.route('/api/list_folders', methods=['GET'])
def list_folders():
    try:
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folders = get_image_folders(images_dir)
        
        return jsonify({
            'success': True,
            'folders': folders
        })
    except Exception as e:
        logger.error(f"Error listing folders: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/list_images', methods=['POST'])
def list_images():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Klasor adi gerekli'
            })
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folder_path = os.path.join(images_dir, folder)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': f'Klasor bulunamadi: {folder}'
            })
        
        images = get_images_in_folder(folder_path)
        
        return jsonify({
            'success': True,
            'folder': folder,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/get_image', methods=['POST'])
def get_image():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder or not filename:
            return jsonify({
                'success': False,
                'message': 'Klasor ve dosya adi gerekli'
            })
        
        filename = secure_filename(filename)
        folder = secure_filename(folder)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        filepath = os.path.join(images_dir, folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        image_url = f"/static/images/{folder}/{filename}"
        
        return jsonify({
            'success': True,
            'url': image_url,
            'filename': filename,
            'folder': folder
        })
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/upload_image', methods=['POST'])
def upload_image():
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
        
        folder = request.form.get('folder', 'uploads')
        folder = secure_filename(folder)
        
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'message': f'Desteklenmeyen dosya formati: {ext}'
            })
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folder_path = os.path.join(images_dir, folder)
        ensure_directory(folder_path)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder_path, filename)
        
        counter = 1
        while os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            new_name = f"{name}_{counter}{ext}"
            filepath = os.path.join(folder_path, new_name)
            counter += 1
        
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Dosya basariyla yuklendi',
            'filename': os.path.basename(filepath),
            'folder': folder,
            'url': f"/static/images/{folder}/{os.path.basename(filepath)}"
        })
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/delete_image', methods=['POST'])
def delete_image():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder or not filename:
            return jsonify({
                'success': False,
                'message': 'Klasor ve dosya adi gerekli'
            })
        
        folder = secure_filename(folder)
        filename = secure_filename(filename)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        filepath = os.path.join(images_dir, folder, filename)
        
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
        logger.error(f"Error deleting image: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/delete_folder', methods=['POST'])
def delete_folder():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Klasor adi gerekli'
            })
        
        folder = secure_filename(folder)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folder_path = os.path.join(images_dir, folder)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': f'Klasor bulunamadi: {folder}'
            })
        
        import shutil
        shutil.rmtree(folder_path)
        
        return jsonify({
            'success': True,
            'message': f'{folder} klasoru silindi'
        })
    except Exception as e:
        logger.error(f"Error deleting folder: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/create_folder', methods=['POST'])
def create_folder():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Klasor adi gerekli'
            })
        
        folder = secure_filename(folder)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folder_path = os.path.join(images_dir, folder)
        
        if os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': f'Klasor zaten var: {folder}'
            })
        
        ensure_directory(folder_path)
        
        return jsonify({
            'success': True,
            'message': f'{folder} klasoru olusturuldu',
            'folder': folder
        })
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/get_image_info', methods=['POST'])
def get_image_info():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder or not filename:
            return jsonify({
                'success': False,
                'message': 'Klasor ve dosya adi gerekli'
            })
        
        folder = secure_filename(folder)
        filename = secure_filename(filename)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        filepath = os.path.join(images_dir, folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        stat = os.stat(filepath)
        
        image_info = {
            'filename': filename,
            'folder': folder,
            'size': stat.st_size,
            'size_str': format_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'url': f"/static/images/{folder}/{filename}"
        }
        
        try:
            from PIL import Image
            img = Image.open(filepath)
            image_info['width'] = img.width
            image_info['height'] = img.height
            image_info['format'] = img.format
            image_info['mode'] = img.mode
        except:
            pass
        
        return jsonify({
            'success': True,
            'info': image_info
        })
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/download_image', methods=['POST'])
def download_image():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder or not filename:
            return jsonify({
                'success': False,
                'message': 'Klasor ve dosya adi gerekli'
            })
        
        folder = secure_filename(folder)
        filename = secure_filename(filename)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        filepath = os.path.join(images_dir, folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': f'Dosya bulunamadi: {filename}'
            })
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@visual_bp.route('/api/batch_delete', methods=['POST'])
def batch_delete():
    try:
        data = request.get_json()
        images = data.get('images', [])
        
        if not images:
            return jsonify({
                'success': False,
                'message': 'Silinecek gorsel secilmedi'
            })
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        deleted = []
        failed = []
        
        for img in images:
            folder = secure_filename(img.get('folder', ''))
            filename = secure_filename(img.get('filename', ''))
            
            if not folder or not filename:
                failed.append({'filename': filename, 'error': 'Gecersiz isim'})
                continue
            
            filepath = os.path.join(images_dir, folder, filename)
            
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    deleted.append(filename)
                except Exception as e:
                    failed.append({'filename': filename, 'error': str(e)})
            else:
                failed.append({'filename': filename, 'error': 'Dosya bulunamadi'})
        
        return jsonify({
            'success': True,
            'deleted': deleted,
            'failed': failed,
            'total': len(images),
            'deleted_count': len(deleted)
        })
    except Exception as e:
        logger.error(f"Error batch deleting: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/rename_image', methods=['POST'])
def rename_image():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        old_name = data.get('old_name', '')
        new_name = data.get('new_name', '')
        
        if not folder or not old_name or not new_name:
            return jsonify({
                'success': False,
                'message': 'Klasor, eski ve yeni dosya adi gerekli'
            })
        
        folder = secure_filename(folder)
        old_name = secure_filename(old_name)
        new_name = secure_filename(new_name)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        old_path = os.path.join(images_dir, folder, old_name)
        new_path = os.path.join(images_dir, folder, new_name)
        
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
        logger.error(f"Error renaming image: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/folder_stats', methods=['POST'])
def folder_stats():
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Klasor adi gerekli'
            })
        
        folder = secure_filename(folder)
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        folder_path = os.path.join(images_dir, folder)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': f'Klasor bulunamadi: {folder}'
            })
        
        images = get_images_in_folder(folder_path)
        total_size = sum(img['size'] for img in images)
        
        return jsonify({
            'success': True,
            'stats': {
                'folder': folder,
                'image_count': len(images),
                'total_size': total_size,
                'total_size_str': format_size(total_size),
                'oldest': images[-1]['modified_str'] if images else None,
                'newest': images[0]['modified_str'] if images else None
            }
        })
    except Exception as e:
        logger.error(f"Error getting folder stats: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@visual_bp.route('/api/search', methods=['POST'])
def search_images():
    try:
        data = request.get_json()
        query = data.get('query', '').strip().lower()
        folder = data.get('folder', '')
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Arama sorgusu gerekli'
            })
        
        images_dir = current_app.config.get('IMAGES_LOG_DIR', 'static/images')
        
        if folder:
            folder = secure_filename(folder)
            folder_path = os.path.join(images_dir, folder)
            if not os.path.exists(folder_path):
                return jsonify({
                    'success': False,
                    'message': f'Klasor bulunamadi: {folder}'
                })
            folders_to_search = [folder_path]
        else:
            folders_to_search = []
            for f in os.listdir(images_dir):
                f_path = os.path.join(images_dir, f)
                if os.path.isdir(f_path):
                    folders_to_search.append(f_path)
        
        results = []
        for f_path in folders_to_search:
            for file in os.listdir(f_path):
                if query in file.lower():
                    file_path = os.path.join(f_path, file)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}:
                            stat = os.stat(file_path)
                            folder_name = os.path.basename(f_path)
                            results.append({
                                'name': file,
                                'folder': folder_name,
                                'size_str': format_size(stat.st_size),
                                'modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                                'url': f"/static/images/{folder_name}/{file}"
                            })
        
        results.sort(key=lambda x: x['name'])
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        logger.error(f"Error searching images: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })