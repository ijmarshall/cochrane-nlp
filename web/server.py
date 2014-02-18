from flask import Flask, request, jsonify
import json
import pprint

app = Flask(__name__)
pp = pprint.PrettyPrinter(indent=4)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/annotate', methods=['POST'])
def annotate():
    text = json.loads(request.data)
    return jsonify({"annotations" : []});

if __name__ == "__main__":
    app.run(debug=True)
