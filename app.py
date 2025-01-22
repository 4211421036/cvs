from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'message': 'Backend server running!',
        'status': 'OK'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
