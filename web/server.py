from flask import Flask, request, jsonify
from pipeline import MockPipeline, RoBPipeline, RegularPipeline, RegularFakeSentPipeline
import pprint
import json
import pickle
import logging

DEBUG_MODE = True

logging.basicConfig(level= (logging.DEBUG if DEBUG_MODE else logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline = RoBPipeline()

pp = pprint.PrettyPrinter(indent=4)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/annotate', methods=['POST'])
def annotate():
    payload = json.loads(request.data)
    result = pipeline.run(payload["pages"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=DEBUG_MODE)
