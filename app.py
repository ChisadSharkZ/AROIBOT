
from flask import Flask, request, jsonify
from flask_cors import CORS
import importlib.util
import os
import sys

# === Import SpeechFlow predict function ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'Spoken_language_identification'))
from Spoken_language_identification import predict_by_pb

# === Import restaurant recommender ===
spec = importlib.util.spec_from_file_location("restaurant_recommender", os.path.join(os.path.dirname(__file__), "restaurant_recommender.py"))
restaurant_recommender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restaurant_recommender)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/voice', methods=['POST'])
def handle_voice():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']

    # === ใช้ SpeechFlow (mock: ใช้ข้อความแทนเสียง) ===
    if any(c >= '\u0E00' and c <= '\u0E7F' for c in text):
        language = 'thai'
    else:
        language = 'english'

    try:
        menu = restaurant_recommender.recommend_menu(text, language)
        response = jsonify({'language': language, 'menu': menu})
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        return jsonify({'error': f'Error in recommend_menu: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
