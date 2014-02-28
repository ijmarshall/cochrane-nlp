from flask import Flask, request, jsonify
from pipeline import MockPipeline, RoBPipeline
import pprint
import json

app = Flask(__name__)
# pipeline = MockPipeline()
pipeline = RoBPipeline()

pp = pprint.PrettyPrinter(indent=4)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/annotate', methods=['POST'])
def annotate():
    payload = json.loads(request.data)

    result = pipeline.run(payload["pages"])
    return jsonify(result);

if __name__ == "__main__":
    app.run(debug=True)
