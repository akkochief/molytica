from flask import Blueprint, render_template, request, jsonify, send_file, current_app
import os
import xml.etree.ElementTree as ET
import json
import traceback
import logging
from datetime import datetime

compare_bp = Blueprint('compare', __name__, url_prefix='/compare-models')
logger = logging.getLogger(__name__)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def parse_xml_models(xml_path, lang='en'):
    models = []
    lang = lang if lang in ('en', 'tr') else 'en'
    other_lang = 'tr' if lang == 'en' else 'en'

    if not os.path.exists(xml_path):
        logger.warning(f"XML file not found: {xml_path}")
        return models

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        model_nodes = root.findall('.//model')

        for node in model_nodes:
            model = {
                'id': node.get('id', f"model_{len(models)+1}"),
                'name': get_node_text(node, 'name', lang, f"Model {len(models)+1}"),
                'description': get_node_text(node, 'description', lang, 'No description available'),
                'detailed_description': get_node_text(node, 'detailed_description', lang, '<p>No detailed information available.</p>'),
                'accuracy': float(get_node_text(node, 'accuracy', lang, '0') or 0),
                'speed': float(get_node_text(node, 'speed', lang, '0') or 0),
                'specs': {},
                'benchmarks': {},
                'pros': [],
                'cons': []
            }

            for spec in node.findall('.//spec'):
                name = spec.get(f'name_{lang}') or spec.get(f'name_{other_lang}') or spec.get('name', 'Specification')
                model['specs'][name] = spec.text.strip() if spec.text else '-'

            for bench in node.findall('.//benchmark'):
                name = bench.get(f'name_{lang}') or bench.get(f'name_{other_lang}') or bench.get('name', 'Benchmark')
                model['benchmarks'][name] = bench.text.strip() if bench.text else '-'

            pro_nodes = node.findall(f'.//pro_{lang}')
            if not pro_nodes:
                pro_nodes = node.findall('.//pro')
            for pro in pro_nodes:
                if pro.text:
                    model['pros'].append(pro.text.strip())

            con_nodes = node.findall(f'.//con_{lang}')
            if not con_nodes:
                con_nodes = node.findall('.//con')
            for con in con_nodes:
                if con.text:
                    model['cons'].append(con.text.strip())

            model['perf'] = [
                model['accuracy'],
                max(0, model['accuracy'] - 2),
                max(0, model['accuracy'] - 4),
                model['speed']
            ]

            models.append(model)

        return models

    except ET.ParseError as e:
        logger.error(f"XML parse error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"XML read error: {str(e)}")
        return []

def get_node_text(parent, tag, lang='en', default=''):
    other_lang = 'tr' if lang == 'en' else 'en'
    node = parent.find(f'{tag}_{lang}')
    if node is not None and node.text:
        return node.text.strip()
    node = parent.find(f'{tag}_{other_lang}')
    if node is not None and node.text:
        return node.text.strip()
    node = parent.find(tag)
    if node is not None and node.text:
        return node.text.strip()
    return default

def create_default_xml(xml_path):
    ensure_directory(os.path.dirname(xml_path))
    
    default_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<models>
    <model id="model1">
        <name>Random Forest</name>
        <description>Ensemble learning with decision trees</description>
        <detailed_description>
            <p>Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and prevent overfitting.</p>
            <h4>Key Features:</h4>
            <ul>
                <li>Handles high-dimensional data well</li>
                <li>Provides feature importance scores</li>
                <li>Robust to outliers and noise</li>
            </ul>
        </detailed_description>
        <accuracy>92.5</accuracy>
        <speed>48.2</speed>
        <specs>
            <spec name="Algorithm">Ensemble of Decision Trees</spec>
            <spec name="Training Time">Fast</spec>
            <spec name="Memory Usage">Medium</spec>
            <spec name="Parallelization">Yes</spec>
        </specs>
        <benchmarks>
            <benchmark name="Validation Accuracy">92.5%</benchmark>
            <benchmark name="F1 Score">0.91</benchmark>
            <benchmark name="AUC-ROC">0.96</benchmark>
        </benchmarks>
        <pro>High accuracy on most datasets</pro>
        <pro>Provides feature importance</pro>
        <pro>Handles missing values well</pro>
        <con>Can be memory intensive</con>
        <con>Less interpretable than single trees</con>
    </model>
    <model id="model2">
        <name>XGBoost</name>
        <description>Gradient boosting with advanced regularization</description>
        <detailed_description>
            <p>XGBoost is an optimized gradient boosting algorithm that uses advanced regularization techniques to prevent overfitting.</p>
            <h4>Key Features:</h4>
            <ul>
                <li>Parallelized tree building</li>
                <li>Built-in regularization</li>
                <li>Handles missing values natively</li>
            </ul>
        </detailed_description>
        <accuracy>94.8</accuracy>
        <speed>42.5</speed>
        <specs>
            <spec name="Algorithm">Gradient Boosting</spec>
            <spec name="Training Time">Moderate</spec>
            <spec name="Memory Usage">Low</spec>
            <spec name="Parallelization">Yes</spec>
        </specs>
        <benchmarks>
            <benchmark name="Validation Accuracy">94.8%</benchmark>
            <benchmark name="F1 Score">0.93</benchmark>
            <benchmark name="AUC-ROC">0.97</benchmark>
        </benchmarks>
        <pro>State-of-the-art performance</pro>
        <pro>Built-in regularization</pro>
        <pro>Efficient memory usage</pro>
        <con>Requires careful hyperparameter tuning</con>
        <con>Less interpretable than Random Forest</con>
    </model>
    <model id="model3">
        <name>Neural Network</name>
        <description>Deep learning with multiple hidden layers</description>
        <detailed_description>
            <p>A multi-layer perceptron neural network with multiple hidden layers capable of learning complex patterns.</p>
            <h4>Key Features:</h4>
            <ul>
                <li>Can learn non-linear relationships</li>
                <li>Flexible architecture</li>
                <li>End-to-end learning</li>
            </ul>
        </detailed_description>
        <accuracy>90.2</accuracy>
        <speed>35.8</speed>
        <specs>
            <spec name="Algorithm">Deep Learning</spec>
            <spec name="Training Time">Slow</spec>
            <spec name="Memory Usage">High</spec>
            <spec name="Parallelization">GPU Support</spec>
        </specs>
        <benchmarks>
            <benchmark name="Validation Accuracy">90.2%</benchmark>
            <benchmark name="F1 Score">0.88</benchmark>
            <benchmark name="AUC-ROC">0.94</benchmark>
        </benchmarks>
        <pro>Can learn complex patterns</pro>
        <pro>Scalable with GPU</pro>
        <pro>Flexible architecture</pro>
        <con>Requires large datasets</con>
        <con>Black box model</con>
        <con>Long training time</con>
    </model>
    <model id="model4">
        <name>LightGBM</name>
        <description>Lightweight gradient boosting with leaf-wise growth</description>
        <detailed_description>
            <p>LightGBM is a gradient boosting framework that uses tree-based learning algorithms and focuses on efficiency.</p>
            <h4>Key Features:</h4>
            <ul>
                <li>Leaf-wise tree growth</li>
                <li>Histogram-based learning</li>
                <li>Fast training speed</li>
            </ul>
        </detailed_description>
        <accuracy>93.1</accuracy>
        <speed>52.3</speed>
        <specs>
            <spec name="Algorithm">Gradient Boosting</spec>
            <spec name="Training Time">Fast</spec>
            <spec name="Memory Usage">Low</spec>
            <spec name="Parallelization">Yes</spec>
        </specs>
        <benchmarks>
            <benchmark name="Validation Accuracy">93.1%</benchmark>
            <benchmark name="F1 Score">0.91</benchmark>
            <benchmark name="AUC-ROC">0.96</benchmark>
        </benchmarks>
        <pro>Very fast training</pro>
        <pro>Low memory usage</pro>
        <pro>High accuracy</pro>
        <con>Can overfit on small datasets</con>
        <con>Leaf-wise growth can be unstable</con>
    </model>
</models>'''
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(default_xml)
    
    logger.info(f"Default XML file created: {xml_path}")

@compare_bp.route('/')
def index():
    xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
    
    if not os.path.exists(xml_path):
        create_default_xml(xml_path)
    
    return render_template('compare.html')

@compare_bp.route('/api/models', methods=['GET'])
def get_models():
    try:
        lang = request.args.get('lang', 'en')
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        models = parse_xml_models(xml_path, lang)
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/model/<model_id>', methods=['GET'])
def get_model(model_id):
    try:
        lang = request.args.get('lang', 'en')
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        models = parse_xml_models(xml_path, lang)
        
        for model in models:
            if model['id'] == model_id:
                return jsonify({
                    'success': True,
                    'model': model
                })
        
        return jsonify({
            'success': False,
            'message': f'Model not found: {model_id}'
        }), 404
    except Exception as e:
        logger.error(f"Model retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/compare', methods=['POST'])
def compare_models():
    try:
        data = request.get_json()
        model1_id = data.get('model1')
        model2_id = data.get('model2')
        lang = data.get('lang', request.args.get('lang', 'en'))
        
        if not model1_id or not model2_id:
            return jsonify({
                'success': False,
                'message': 'Two model IDs are required'
            })
        
        if model1_id == model2_id:
            return jsonify({
                'success': False,
                'message': 'Cannot compare the same model'
            })
        
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        models = parse_xml_models(xml_path, lang)
        
        model1 = None
        model2 = None
        
        for model in models:
            if model['id'] == model1_id:
                model1 = model
            if model['id'] == model2_id:
                model2 = model
        
        if not model1 or not model2:
            return jsonify({
                'success': False,
                'message': 'One of the models could not be found'
            })
        
        comparison = {
            'model1': model1,
            'model2': model2,
            'comparison': {
                'accuracy_winner': model1['name'] if model1['accuracy'] > model2['accuracy'] else (model2['name'] if model2['accuracy'] > model1['accuracy'] else 'tie'),
                'speed_winner': model1['name'] if model1['speed'] > model2['speed'] else (model2['name'] if model2['speed'] > model1['speed'] else 'tie'),
                'accuracy_diff': abs(model1['accuracy'] - model2['accuracy']),
                'speed_diff': abs(model1['speed'] - model2['speed'])
            }
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/save_xml', methods=['POST'])
def save_xml():
    try:
        data = request.get_json()
        xml_content = data.get('xml', '')
        
        if not xml_content:
            return jsonify({
                'success': False,
                'message': 'XML content is empty'
            })
        
        try:
            ET.fromstring(xml_content)
        except ET.ParseError as e:
            return jsonify({
                'success': False,
                'message': f'Invalid XML: {str(e)}'
            })
        
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        ensure_directory(os.path.dirname(xml_path))
        
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        return jsonify({
            'success': True,
            'message': 'XML saved successfully'
        })
    except Exception as e:
        logger.error(f"XML save error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/get_xml', methods=['GET'])
def get_xml():
    try:
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'success': True,
            'xml': content
        })
    except Exception as e:
        logger.error(f"XML retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/export_comparison', methods=['POST'])
def export_comparison():
    try:
        data = request.get_json()
        model1 = data.get('model1', {})
        model2 = data.get('model2', {})
        format_type = data.get('format', 'json')
        
        if not model1 or not model2:
            return jsonify({
                'success': False,
                'message': 'Comparison data is required'
            })
        
        comparison_data = {
            'exported_at': datetime.now().isoformat(),
            'model1': model1,
            'model2': model2,
            'comparison': {
                'accuracy_winner': model1['name'] if model1.get('accuracy', 0) > model2.get('accuracy', 0) else (model2['name'] if model2.get('accuracy', 0) > model1.get('accuracy', 0) else 'tie'),
                'speed_winner': model1['name'] if model1.get('speed', 0) > model2.get('speed', 0) else (model2['name'] if model2.get('speed', 0) > model1.get('speed', 0) else 'tie')
            }
        }
        
        if format_type == 'json':
            return jsonify({
                'success': True,
                'content': json.dumps(comparison_data, indent=2, ensure_ascii=False),
                'format': 'json'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Unsupported format: {format_type}'
            })
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/update_model', methods=['POST'])
def update_model():
    try:
        data = request.get_json()
        model_id = data.get('id')
        updates = data.get('updates', {})
        
        if not model_id:
            return jsonify({
                'success': False,
                'message': 'Model ID is required'
            })
        
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        model_node = root.find(f".//model[@id='{model_id}']")
        
        if model_node is None:
            return jsonify({
                'success': False,
                'message': f'Model not found: {model_id}'
            })
        
        for key, value in updates.items():
            node = model_node.find(key)
            if node is not None:
                node.text = str(value)
            else:
                new_node = ET.SubElement(model_node, key)
                new_node.text = str(value)
        
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        return jsonify({
            'success': True,
            'message': f'Model {model_id} updated'
        })
    except Exception as e:
        logger.error(f"Model update error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/reload', methods=['POST'])
def reload_models():
    try:
        lang = request.args.get('lang') or (request.get_json(silent=True) or {}).get('lang', 'en')
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        models = parse_xml_models(xml_path, lang)
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models),
            'message': 'Models reloaded'
        })
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@compare_bp.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        lang = request.args.get('lang', 'en')
        xml_path = os.path.join(current_app.root_path, 'static', 'xml_data', 'data.xml')
        
        if not os.path.exists(xml_path):
            create_default_xml(xml_path)
        
        models = parse_xml_models(xml_path, lang)
        
        if not models:
            return jsonify({
                'success': True,
                'stats': {
                    'total_models': 0,
                    'avg_accuracy': 0,
                    'avg_speed': 0,
                    'best_accuracy': 0,
                    'best_speed': 0,
                    'best_accuracy_model': '',
                    'best_speed_model': ''
                }
            })
        
        total = len(models)
        avg_accuracy = sum(m['accuracy'] for m in models) / total
        avg_speed = sum(m['speed'] for m in models) / total
        
        best_acc = max(models, key=lambda m: m['accuracy'])
        best_speed = max(models, key=lambda m: m['speed'])
        
        return jsonify({
            'success': True,
            'stats': {
                'total_models': total,
                'avg_accuracy': round(avg_accuracy, 2),
                'avg_speed': round(avg_speed, 2),
                'best_accuracy': best_acc['accuracy'],
                'best_speed': best_speed['speed'],
                'best_accuracy_model': best_acc['name'],
                'best_speed_model': best_speed['name']
            }
        })
    except Exception as e:
        logger.error(f"Statistics retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })