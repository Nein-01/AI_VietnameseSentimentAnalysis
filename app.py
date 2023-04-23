# encoding: utf-8
import datetime

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from config import *
import json
import numpy as np

app = Flask(__name__)
model = load_model(CLS_MODEL)
def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/api/sender', methods=['POST'])
def receive_text():
    content_type = request.headers.get('Content-Type')
    restData=''
    if (content_type == 'application/json'):
        jsonData = request.json
        print(jsonData['data'])
        restData = jsonData['data']
        #return json
    else:
        #return 'Content-Type not supported!'
        return content_type+' is not supported!'
    #text = request.form['text']
    sentiment = predict_sentiment(restData)
    print(restData)
    print(sentiment)
    return jsonify({'result': myconverter(sentiment)})

def predict_sentiment(text):
    if text != '':
        if NN_ARCHITECTURE == 'cnn':
            from cnn import CNN_predict
            return CNN_predict(text, model)
        elif NN_ARCHITECTURE == 'lstm-cnn':
            from lstm_cnn import LSTM_CNN_predict
            return LSTM_CNN_predict(text, model)
    return -1

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug=True)