from utils.dir import create_dirs
import cv2
import numpy as np
import collections
import tensorflow as tf
import sklearn
import os
import sys
import math
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import scipy
import imageio
import time
from time import sleep
from utils import (
    load_model,
    get_faces_live,
    forward_pass,
    load_embeddings,
    save_image
)


# def detect_face_frames(stream):
#     camera = cv2.VideoCapture(stream)
#     input_frame=0
#     output_frame=0

#     try:
#         input_path, output_path = create_dirs('./testing_apis/face_detection/test-%s')
#     except Exception as e:
#         print('\n\n\nerror: ', e)


#     while True:
#         success, frame = camera.read()
#         print('\n\n\nreading\n\n\n')
#         if not success:
#             break
#         else:
#             print('\n\n\n going to model \n\n\n')
#             cv2.imwrite('./input/i'+ str(input_frame)+'.jpg', frame)
#             input_frame=input_frame+1
#             #frame = run_facenet_model(frame)
#             cv2.imwrite('./output/o'+str(output_frame)+'.jpg', frame)
#             output_frame=output_frame+1
#             frame = frame.tobytes()
#             yield(b'--frame\r\n'
#                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_face_frames(stream):

    print('Loading feature extraction model')
    FACENET_MODEL_PATH = PATH+'/model/20170512-110547/20170512-110547.pb'
    facenet_model=load_model(FACENET_MODEL_PATH)

    CLASSIFIER_PATH = PATH+'/model/best_model2.pkl'
    OUT_ENCODER_PATH = PATH+'/model/out_encoder.pkl'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    # Load the out encoder 
    with open(OUT_ENCODER_PATH, 'rb') as file:
        out_encoder = pickle.load(file)
    print("Out Encoder, Successfully loaded")

    people_detected = set()
    person_detected = collections.Counter()

    image_size = 160
    tf.compat.v1.disable_eager_execution()
    config = tf.compat.v1.ConfigProto(device_count={'GPU':0, 'CPU':1})

    # Get input and output tensors
    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    # Initiate persistent FaceNet model in memory
    facenet_persistent_session = tf.compat.v1.Session(graph=facenet_model, config=config, log_device_placement=True))

    # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None) 
    camera = cv2.VideoCapture(stream)
    # camera = camera.set(cv2.CV_CAP_PROP_FPS, 30)
    input_frame = 0
    output_frame = 0

    try:
        input_path, output_path = create_dirs('./testing_apis/face_detection/test-%s')
    except Exception as e:
        print('\n\n\nerror: ', e)


    while camera.isOpened():
        success, frame_orig = camera.read()
        if not success:
            break
        else:
            # Resize frame to half its size for faster computation
            frame = cv2.resize(src=frame_orig, dsize=(0, 0), fx=0.5, fy=0.5)
            try:
                cv2.imwrite(input_path + '/i'+str(input_frame)+'.jpg', frame)
                input_frame+=1
             
                faces, rects = get_faces_live(
                        img=frame,
                        pnet=pnet,
                        rnet=rnet,
                        onet=onet,
                        image_size=image_size
                    )

                # If there are human faces detected
                if faces:
                    for i in range(len(faces)):
                        face_img = faces[i]
                        rect = rects[i]

                        # Scale coordinates of face locations by the resize ratio
                        rect = [coordinate * 2 for coordinate in rect]

                        face_embedding = forward_pass(
                            mg=face_img,
                            session=facenet_persistent_session,
                            images_placeholder=images_placeholder,
                            embeddings=embeddings,
                            phase_train_placeholder=phase_train_placeholder,
                            image_size=image_size
                        )

                        print("\t\t\t\t\t\t\t\n\n\n\n\n\n")
                        print("EMBEDDINGS SHAPE: ", face_embedding.shape)
                        print("\n\n\n\\n\n\n")

                        predictions = model.predict_proba(face_embedding)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = out_encoder.inverse_transform(best_class_indices)
                        best_name = best_name[0]
                        print("\n\n\n Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20  
                        
                        if best_class_probabilities > 0.25:
                            name = best_name
                        else:
                            name = "Unknown"
                        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                        cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)


                cv2.imwrite(output_path + '/o'+str(output_frame)+'.jpg', frame)
                output_frame+=1
                frame = cv2.imencode('.jpeg', frame)[1]
                frame = frame.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print('\n\n\nerror: ', e)
                break