import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from keras_preprocessing import image
import cv2
from PIL import Image

# has to be declared globally in app.py
# follow this https://blog.paperspace.com/deploying-deep-learning-models-flask-web-python/
global graph, model
model, graph = init()

# this entire function has to be used in the route post in app.py
def predict(frame):
    # Predict method is called when we push the 'Predict' button 
    # on the webpage. We will feed the user drawn image to the model
    # perform inference, and return the classification

    #this has to be loaded in the main of app.py as model should be loaded once not for every frame of the video
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('FSD/fsd_model.h5')
    # Show the model architecture
    model.summary()

    #Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')
    #Resizing into 224x224 because we trained the model with this image size.
    im = im.resize((224,224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    
    with graph.as_default():
        probabilities = model.predict(img_array)
		print(probabilities)
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        print(prediction)
        #if prediction is 0, which means there is fire in the frame.
        if (prediction != 1) :
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            print(probabilities[prediction])
            print('Alarm Situation')
		return prediction
    

