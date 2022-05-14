from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import os
import io
import sys
import numpy as np
import base64
from generators.face_detection_generator import detect_face_frames
from time import sleep

app = Flask(__name__)

@app.route('/detectFace', methods=['GET', 'POST'])
def detect_face():
    stream = './test_images/acs_test.mp4'
    return Response(detect_face_frames(stream), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faceTemplate')
def home():
    return render_template('face.html')

@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=False, port = 5000)
