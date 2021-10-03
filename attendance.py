# necessary Import for Facial recognition
import cv2
import numpy as np
import face_recognition  #this import is based on dilib :: what exactly dilib does is it basically finds 128 unique points from human face
import os
from datetime import datetime # as we will be needing both date and time to display along with the name of the
# necessary import for Facial Expression Detection 


path = 'images'  # Set path for backend image folder
images = [] # creating a list with name image jiske andar hamari sari ki sari image chali jayegi 
personNames = []  # creating a list with name personName jiske andar hamari sari ki sari image Name chali jayegi  chali jayegi 
myList = os.listdir(path) #current directory ki sare components ki listing hogi
print(myList)
# Seperating image name from its extended version 
# for eg. biden.jpg => biden
# for loop is used here to traverse through the array 
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')   #here cu_img is storing the index and read command is helping in reading the image
    images.append(current_Img) # basically we are adding the readed image in our list
    personNames.append(os.path.splitext(cu_img)[0]) # seperating image name with its extension here image name is at index 0 and its extension is at index 1::: thats exactly how python treats the name of the file with its extension 
print(personNames)
# Below function fetch the 128 unique features from a human face
# To find encoding "HOG algorythm is being used"
def faceEncodings(images):
    encodeList = [] # creating list to save 128 features of face 
    for img in images: # traversing through list of the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # as these images gets readed through cv2 it is basically in BGR formate so we first we have to convirt the above given image in RGB.
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode) # adding data in encodeList[] array
    return encodeList

#function to mark attendance
def attendance(name):
    with open('Attendance.csv', 'r+') as f: #opened csv file in read and append mode
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S') #fetch time
            dStr = time_now.strftime('%d/%m/%Y') #fetch date
            f.writelines(f'\n{name},{tStr},{dStr}') #pass data to csv file


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0) #To read video from camera :: by default value for inbuilt camera is 0 and external is 1. 

while True:   # its a infinite loop to read video from camera
    ret, frame = cap.read() #to read camera ret as well as frame 
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) #resizing the frame by 1/4 og the original size that is 0.25 
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB) # again BGR to RGB conversion 

    facesCurrentFrame = face_recognition.face_locations(faces) #finding fac Location in live video
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame) #encoding faces from current frame or LIve video
# To check whether encoded current frame face image is same or not, By comparing it with original image database 
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # comparing image with encoded video
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #Finding facdis if the facedis is less then image is matched
        # print(faceDis)
        matchIndex = np.argmin(faceDis) #finding index value for facdis 

        if matches[matchIndex]: #finally comparing encoded image at matchIndex position image 
            name = personNames[matchIndex].upper() # fetching the name from personNames 
            # print(name)
            # creating square around image 
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 #resizing to original size 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #deciding color and rectangle
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # putting name of the person over rectangular
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13: #ASCII code for enter button is 13 :: specifying key to stop camera . 
        break

cap.release() #start capturing video
cv2.destroyAllWindows()
