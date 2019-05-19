#Author : Shaik Masihullah


# Loading the trained model
from keras.models import load_model
classifier = load_model('.Trained_model/emotion_little_vgg_new.h5')

# Defining the class labels for prediction
class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

# Importing required dependencies 
import cv2
import numpy as np
from time import sleep
from keras.preprocessing.image import img_to_array
import random

def face_detector(img):
    """
    Returns the detected cropped face after drawing a rectangle box around the face
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img

def draw_test(im,pred,x,y,w,h):
    """
    Returns the image after adding the respective emoji of the detected emotion
    """

    img1 = im
    rows,cols,channels = img1.shape
    
    if w<=rows and h<=cols:
        w = w 
        h = h
    else:
        w = rows
        h = cols

    img2 = cv2.resize(cv2.imread('./emojis/'+ pred + '.png'),(w,h))
    
    roi = img1[y:y+h, x:x+w ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[y:y+h, x:x+w ] = dst

    return img1  


def game(boolean,flag,time,image,label,count,fail):
    """
    Made a fun game using emotions
    """
    label_position = (20, 50)
    label_position1 = (20, 90)
    if boolean == True:
        r = random.randint(0,5)
        label = class_labels[r]
        count = count + 1
        label_position = (2, 2)
        flag = True
        cv2.putText(image, 'Try being ' + label + '?', label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3)
    elif time < 25:
        cv2.putText(image, 'Try being ' + label + '?', label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3)
    else:
        cv2.putText(image, 'Try being ' + label + '?', label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3)
        cv2.putText(image, 'Wrong, Try again!', label_position1 , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3)
        if flag == True:
            fail = fail + 1
            flag = False
        
    return image,label,count,fail,flag
        


face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

boolean = True
q = class_labels[3]
time = 0
count = 0
fail = 0
flag = False
while True:
    time = time + 1
    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    BLACK = [0,0,0]
    image = cv2.copyMakeBorder(image,300, 0, 0,0 ,cv2.BORDER_CONSTANT,value=BLACK)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        image,q,count,fail,flag = game(boolean,flag,time,image,q,count,fail)


        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]  
        predicted = draw_test(image,label,rect[0],rect[1],rect[2],rect[3])
        if q == label:
            boolean = True
            time = 0
        else:
            boolean = False
    else:
        predicted = image
        cv2.putText(predicted, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

    cv2.imshow('Emojinator', predicted)
    if cv2.waitKey(1) == 13 or count == 10: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()    
print('Your Score : ' + str(count-fail) + '/' + str(count)) 
