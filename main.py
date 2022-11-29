from flask import *
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from time import gmtime, strftime
#import sys
import os
import sys
import cv2
from pyngrok import ngrok, conf
#import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('style.xml')
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(21, 28)', '(30, 40)', '(44, 54)', '(55, 75)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

age_net, gender_net = load_caffe_models()
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def init_webhooks(base_url):
    pass

app = Flask(__name__)
public_url=ngrok.connect(5000)
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, 5000))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        #result = image_processing(file_path)
        frame = cv2.imread(file_path)
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        result=""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(emotion_dict[maxindex])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)
            # Get Face as Matrix and copy it
            face_img = frame[y:y + h, h:h + w].copy()
            #print(face_img)
            blob=cv2.dnn.blobFromImage(face_img,1,(227,227),MODEL_MEAN_VALUES,swapRB=False)#**
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "%s %s" % (gender, age)
            print(overlay_text)
            if age=="(0, 2)":
                result="Age is "+str(age)
            if age=="(4, 6)":
                result="Age is "+str(age)
            if age=="(8, 12)":
                result="Age is "+str(age)
            if age=="(15, 20)":
                result="Age is "+str(age)
            if age=="(21, 28)":
                result="Age is "+str(age)
            if age=="(30, 40)":
                result="Age is "+str(age)
            if age=="(44, 54)":
                result="Age is "+str(age)
            if age=="(55, 75)":
                result="Age is "+str(age)
                
            if gender=="Male":
                result+="\n Gender is "+str(gender)
            if gender=="Female":
                result+="\n Gender is "+str(gender)
            
            if emotion_dict[maxindex]=="Angry" and age=="(0, 2)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Eat or Drink "
            if emotion_dict[maxindex]=="Angry" and age=="(4, 6)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Go out and play \n Movie Suggestion: Chota Bheem"
            if emotion_dict[maxindex]=="Angry" and age=="(8, 12)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Eat ice cream \n Movie Suggestion: Mr. and Mrs. Smith"
            if emotion_dict[maxindex]=="Angry" and age=="(15, 20)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Drink water and walk it off \n Movie Suggestion: Pursuit of Happyness"
            if emotion_dict[maxindex]=="Angry" and age=="(21, 28)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Eat dark chocolate \n Movie Suggestion: KGF Chapter 1 & 2, Baahubali 1 & 2"
            if emotion_dict[maxindex]=="Angry" and age=="(30, 40)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Consume hot beverage \n Movie Suggestion: Jathi Rathnalu"
            if emotion_dict[maxindex]=="Angry" and age=="(44, 54)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Do meditation \n Movie Suggestion: Devotional Movies"
            if emotion_dict[maxindex]=="Angry" and age=="(55, 75)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Breathing exercises \n Movie Suggestion: Devotional Movies"
            
            
            if emotion_dict[maxindex]=="Disgusted":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Do breathing exercises"

                
            if emotion_dict[maxindex]=="Fearful":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Close your eyes and think positive "

                
            if emotion_dict[maxindex]=="Happy" and age=="(0, 2)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Listen to rhymes \n Movie Suggestion: Monions, Shrek  "
            if emotion_dict[maxindex]=="Happy" and age=="(4, 6)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Watch Catroon \n Movie Suggestion: Mr. Bean, Spider-Man  "
            if emotion_dict[maxindex]=="Happy" and age=="(8, 12)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Play Outside \n Movie Suggestion: Harry Potter, Twilight  "
            if emotion_dict[maxindex]=="Happy" and age=="(15, 20)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Workout \n Movie Suggestion: Friends, Big Bang Theory  "
            if emotion_dict[maxindex]=="Happy" and age=="(21, 28)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Learn cooking \n Movie Suggestion: Jersey, Uri  "
            if emotion_dict[maxindex]=="Happy" and age=="(30, 40)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Read a book \n Movie Suggestion: KKKG, DDLJ  "
            if emotion_dict[maxindex]=="Happy" and age=="(44, 54)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Go for a walk \n Movie Suggestion: Devotional Movies  "
            if emotion_dict[maxindex]=="Happy" and age=="(55, 75)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Watch TV \n Movie Suggestion: Devotional Movies"
                
            if emotion_dict[maxindex]=="Neutral":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Take a nap."

                
            if emotion_dict[maxindex]=="Sad" and age=="(0, 2)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Drink milk \n Movie Suggestion: Youtube rhymes"
            if emotion_dict[maxindex]=="Sad" and age=="(4, 6)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Play outside \n Movie Suggestion: Doremon"
            if emotion_dict[maxindex]=="Sad" and age=="(8, 12)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Listen to songs \n Movie Suggestion: Avengers"
            if emotion_dict[maxindex]=="Sad" and age=="(15, 20)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Talk to your friend \n Movie Suggestion: Hangover 1, 2 & 3"
            if emotion_dict[maxindex]=="Sad" and age=="(21, 28)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Go for a drive \n Movie Suggestion: Wake Up Sid"
            if emotion_dict[maxindex]=="Sad" and age=="(30, 40)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Go for a walk \n Movie Suggestion: DDLJ, 3 Idiots"
            if emotion_dict[maxindex]=="Sad" and age=="(44, 54)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Go for a walk \n Movie Suggestion: Devotional Movies"
            if emotion_dict[maxindex]=="Sad" and age=="(55, 75)":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\n Suggestion : Listen to Old Songs \n Movie Suggestion: Old Movies"
                
            if emotion_dict[maxindex]=="Surprised":
                result+="\n Emotion is "+emotion_dict[maxindex]+"\nSuggestion : Live the moment."


                

        if not result:
            result="No face recognised in the provided image"
        return result
    return None

if __name__ == '__main__':
    app.run()
