from flask import Blueprint, render_template, request, jsonify, current_app
import os
import logging
import json
from datetime import datetime

help_bp = Blueprint('help', __name__, url_prefix='/help')
logger = logging.getLogger(__name__)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

@help_bp.route('/')
def index():
    return render_template('help.html')

@help_bp.route('/api/faq', methods=['GET'])
def get_faq():
    try:
        faq_data = {
            'tr': [
                {
                    'question': 'Hangi veri boyutlarini isleyebilir?',
                    'answer': '10.000 satira kadar veriyi sorunsuz isleyebilir.'
                },
                {
                    'question': 'Sonuclari nasil disa aktarabilirim?',
                    'answer': 'Her analiz sonucunda "Indir" butonu ile CSV veya PDF olarak kaydedebilirsiniz.'
                },
                {
                    'question': 'Ozel modeller ekleyebilir miyim?',
                    'answer': 'Gelistirici modu ile kendi modellerinizi entegre edebilirsiniz.'
                },
                {
                    'question': 'Veri setimde eksik degerler varsa ne yapmaliyim?',
                    'answer': 'Eksik degerleri doldurun veya satirlari temizleyin. Sistem otomatik olarak eksik degerleri isleyebilir, ancak temiz veri daha iyi sonuc verir.'
                },
                {
                    'question': 'Model egitimi ne kadar surer?',
                    'answer': 'Veri boyutuna bagli olarak 1-5 dakika arasinda surebilir. Buyuk veri setlerinde bu sure uzayabilir.'
                }
            ],
            'en': [
                {
                    'question': 'What data sizes can be processed?',
                    'answer': 'Up to 10,000 rows without issues.'
                },
                {
                    'question': 'How to export results?',
                    'answer': 'Use the "Download" button to save as CSV or PDF.'
                },
                {
                    'question': 'Can I add custom models?',
                    'answer': 'Yes, with developer mode you can integrate your own models.'
                },
                {
                    'question': 'What should I do if my dataset has missing values?',
                    'answer': 'Fill missing values or clean the rows. The system can handle missing values automatically, but clean data yields better results.'
                },
                {
                    'question': 'How long does model training take?',
                    'answer': 'It can take 1-5 minutes depending on data size. Larger datasets may take longer.'
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'faq': faq_data
        })
    except Exception as e:
        logger.error(f"FAQ getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/contact', methods=['GET'])
def get_contact_info():
    try:
        contact_info = {
            'email': 'sefaakkoc@outlook.com',
            'github': 'https://github.com/sefaakkoc',
            'twitter': '#',
            'linkedin': '#'
        }
        
        return jsonify({
            'success': True,
            'contact': contact_info
        })
    except Exception as e:
        logger.error(f"Iletisim bilgisi getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/requirements', methods=['GET'])
def get_requirements():
    try:
        requirements = {
            'tr': {
                'browser': 'Google Chrome, Mozilla Firefox, Safari veya Microsoft Edge (en son surumler)',
                'os': 'Windows 10+, macOS 10.15+ veya Linux (Ubuntu 20.04+)',
                'python': 'Python 3.8+',
                'node': 'Node.js 14+',
                'internet': 'Stabil internet baglantisi'
            },
            'en': {
                'browser': 'Latest versions of Google Chrome, Mozilla Firefox, Safari, or Microsoft Edge',
                'os': 'Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)',
                'python': 'Python 3.8+',
                'node': 'Node.js 14+',
                'internet': 'Stable internet connection'
            }
        }
        
        return jsonify({
            'success': True,
            'requirements': requirements
        })
    except Exception as e:
        logger.error(f"Gereksinim getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/file_formats', methods=['GET'])
def get_file_formats():
    try:
        formats = {
            'tr': [
                {
                    'name': 'CSV',
                    'description': 'Tabular veriler icin standart format. Her kolon basligi duzgun etiketlenmelidir.',
                    'icon': 'fa-file-csv'
                },
                {
                    'name': 'Excel (.xlsx, .xls)',
                    'description': 'Excel dosyalarini CSV\'ye donusturerek yukleyebilirsiniz.',
                    'icon': 'fa-file-excel'
                },
                {
                    'name': 'Manuel Veri Girisi',
                    'description': 'Kucuk veri setleri icin dogrudan veri girisi yapabilirsiniz.',
                    'icon': 'fa-keyboard'
                }
            ],
            'en': [
                {
                    'name': 'CSV',
                    'description': 'Standard format for tabular data. Each column must be properly labeled.',
                    'icon': 'fa-file-csv'
                },
                {
                    'name': 'Excel (.xlsx, .xls)',
                    'description': 'Convert Excel files to CSV format for upload.',
                    'icon': 'fa-file-excel'
                },
                {
                    'name': 'Manual Data Entry',
                    'description': 'Direct data input for small datasets.',
                    'icon': 'fa-keyboard'
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'formats': formats
        })
    except Exception as e:
        logger.error(f"Dosya formati getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/tutorials', methods=['GET'])
def get_tutorials():
    try:
        tutorials = {
            'tr': [
                {
                    'title': 'Baslangic Rehberi',
                    'description': 'Uygulamaya ilk adim ve temel islemler',
                    'url': '#',
                    'duration': '5 dk'
                },
                {
                    'title': 'Veri Yukleme ve Analiz',
                    'description': 'Veri yukleme, temizleme ve analiz islemleri',
                    'url': '#',
                    'duration': '10 dk'
                },
                {
                    'title': 'Tahmin ve Gorsellestirme',
                    'description': 'Tahmin yapma ve sonuclari gorsellestirme',
                    'url': '#',
                    'duration': '8 dk'
                }
            ],
            'en': [
                {
                    'title': 'Getting Started Guide',
                    'description': 'First steps and basic operations',
                    'url': '#',
                    'duration': '5 min'
                },
                {
                    'title': 'Data Upload and Analysis',
                    'description': 'Data upload, cleaning, and analysis',
                    'url': '#',
                    'duration': '10 min'
                },
                {
                    'title': 'Prediction and Visualization',
                    'description': 'Making predictions and visualizing results',
                    'url': '#',
                    'duration': '8 min'
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'tutorials': tutorials
        })
    except Exception as e:
        logger.error(f"Egitim getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/dataset_requirements', methods=['GET'])
def get_dataset_requirements():
    try:
        requirements = {
            'tr': [
                {
                    'title': 'Zaman Bilgisi',
                    'description': 'Zaman serisi tahmini yapiyorsaniz, veriniz tarih/saat icermelidir.'
                },
                {
                    'title': 'Kategorik Ozellikler',
                    'description': 'Kimyasal tur, bolge gibi kategorik degiskenler bulunmalidir.'
                },
                {
                    'title': 'Sayisal Ozellikler',
                    'description': 'Sicaklik, yogunluk, konsantrasyon gibi sayisal degiskenler gereklidir.'
                },
                {
                    'title': 'Eksik Veriler',
                    'description': 'Eksik veriler temizlenmeli veya uygun sekilde doldurulmalidir.'
                },
                {
                    'title': 'Kolon Basliklari',
                    'description': 'Tum kolonlar acik ve anlasilir sekilde etiketlenmelidir.'
                }
            ],
            'en': [
                {
                    'title': 'Time Information',
                    'description': 'For time series forecasting, include date/time data.'
                },
                {
                    'title': 'Categorical Features',
                    'description': 'Include categorical variables like chemical type, region.'
                },
                {
                    'title': 'Numeric Features',
                    'description': 'Include numeric variables like temperature, concentration.'
                },
                {
                    'title': 'Missing Data',
                    'description': 'Clean missing data or fill appropriately.'
                },
                {
                    'title': 'Column Headers',
                    'description': 'All columns must be clearly labeled.'
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'requirements': requirements
        })
    except Exception as e:
        logger.error(f"Veri seti gereksinimi getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/search', methods=['POST'])
def search_help():
    try:
        data = request.get_json()
        query = data.get('query', '').strip().lower()
        lang = data.get('lang', 'tr')
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Arama sorgusu gerekli'
            })
        
        help_content = {
            'tr': [
                'Sistem Gereksinimleri',
                'Desteklenen Dosya Formatlari',
                'Manuel Veri Girisi',
                'CSV Dosyasi Yukle',
                'Excel\'den CSV\'ye Donustur',
                'Veri Analizi',
                'Katalizor Tahmini',
                'Verim Tahmini',
                'Tahmin Yap',
                'Gorsellestirme',
                'Model Karsilastirmasi',
                'Raporlama',
                'Zaman Bilgisi',
                'Kategorik Ozellikler',
                'Sayisal Ozellikler',
                'Eksik Veriler',
                'Kolon Basliklari',
                'Veri boyutlari',
                'Sonuclari disa aktarma',
                'Ozel modeller'
            ],
            'en': [
                'System Requirements',
                'Supported File Formats',
                'Manual Data Entry',
                'Upload CSV File',
                'Convert Excel to CSV',
                'Data Analysis',
                'Catalyst Prediction',
                'Yield Prediction',
                'Make Prediction',
                'Visualization',
                'Model Comparison',
                'Reporting',
                'Time Information',
                'Categorical Features',
                'Numeric Features',
                'Missing Data',
                'Column Headers',
                'Data sizes',
                'Export results',
                'Custom models'
            ]
        }
        
        results = []
        for item in help_content.get(lang, []):
            if query in item.lower():
                results.append(item)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query
        })
    except Exception as e:
        logger.error(f"Arama hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        name = data.get('name', '')
        email = data.get('email', '')
        message = data.get('message', '')
        rating = data.get('rating', 0)
        
        if not message:
            return jsonify({
                'success': False,
                'message': 'Mesaj gerekli'
            })
        
        feedback_dir = os.path.join(current_app.root_path, 'static', 'feedback')
        ensure_directory(feedback_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"feedback_{timestamp}.json"
        filepath = os.path.join(feedback_dir, filename)
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'email': email,
            'message': message,
            'rating': rating
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Geri bildiriminiz icin tesekkurler!'
        })
    except Exception as e:
        logger.error(f"Geri bildirim hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/version', methods=['GET'])
def get_version():
    try:
        version_info = {
            'app': 'Molytica Chemical Analyzer',
            'version': '2.1.0',
            'release_date': '2025-01-15',
            'python_version': '3.8+',
            'flask_version': '2.3.0'
        }
        
        return jsonify({
            'success': True,
            'version': version_info
        })
    except Exception as e:
        logger.error(f"Versiyon getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = {
            'tr': {
                'total_help_articles': 12,
                'total_faq': 5,
                'total_tutorials': 3,
                'supported_formats': 3
            },
            'en': {
                'total_help_articles': 12,
                'total_faq': 5,
                'total_tutorials': 3,
                'supported_formats': 3
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Istatistik getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@help_bp.route('/api/sitemap', methods=['GET'])
def get_sitemap():
    try:
        sitemap = {
            'tr': [
                {'title': 'Ana Sayfa', 'url': '/'},
                {'title': 'Veri Yükleme', 'url': '/dataset'},
                {'title': 'Manuel Veri Girişi', 'url': '/manual'},
                {'title': 'CSV Yükle', 'url': '/upload-csv'},
                {'title': 'Excel\'den CSV', 'url': '/excel_to_csv'},
                {'title': 'Tahmin', 'url': '/predict'},
                {'title': 'ML Tahmin', 'url': '/predict_ml'},
                {'title': 'Görselleştirme', 'url': '/visual'},
                {'title': 'Model Karşılaştırma', 'url': '/compare-models'},
                {'title': 'Yardım', 'url': '/help'}
            ],
            'en': [
                {'title': 'Home', 'url': '/'},
                {'title': 'Data Upload', 'url': '/dataset'},
                {'title': 'Manual Data Entry', 'url': '/manual'},
                {'title': 'Upload CSV', 'url': '/upload-csv'},
                {'title': 'Excel to CSV', 'url': '/excel_to_csv'},
                {'title': 'Prediction', 'url': '/predict'},
                {'title': 'ML Prediction', 'url': '/predict_ml'},
                {'title': 'Visualization', 'url': '/visual'},
                {'title': 'Model Comparison', 'url': '/compare-models'},
                {'title': 'Help', 'url': '/help'}
            ]
        }
        
        return jsonify({
            'success': True,
            'sitemap': sitemap
        })
    except Exception as e:
        logger.error(f"Site haritasi getirme hatasi: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })