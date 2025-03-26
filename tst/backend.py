from flask import Flask, jsonify
import sqlite3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Endpoint to fetch data
@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect('/app/financial_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM stock_data')
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to a list of dictionaries
    data = [{'Date': row[0], 'Open': row[1], 'High': row[2], 'Low': row[3], 'Close': row[4], 'Volume': row[5]} for row in rows]
    return jsonify(data)

# if __name__ == '__main__':
#     app.run(debug=True)