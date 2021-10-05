import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from deepface import DeepFace
path="images"
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
# Import images that are present in myList
for cl in myList:
    curImg=cv2.imread(f'{path}//{cl}')     #reading the image
    images.append(curImg)    
    classNames.append(os.path.splitext(cl)[0])   # removing .jpg from images and appending it to classNames
print(classNames)
#Creating func to find the encodings of the images
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)     #Convert BGR to RGB
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendance(name,results):
     with open('Attendance.csv','r+') as f:   # Opening cvs file
        myDataList = f.readlines()
        nameList=[]    # List containing Names 
        for line in myDataList:
            entry=line.split(',')    
            nameList.append(entry[0])  # Appending names to nameList
            
            nameList.append(results)
         
        
        # If the name is not present in nameList then append it to the nameList
        if name not in nameList:
            now=datetime.now()    #Obtaining the current time
            dtString=now.strftime('%H:%M:%S') #Time in Hour:Minute:Second
            dpstring=now.strftime('%d/%m/%Y') #fetch date
            f.writelines(f'\n{name},{dpstring}{dtString},{results}')
encodeListKnown=findEncodings(images)

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') # Used to detect Faces
#Intialize Webcam
cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    
    try:
     
     result=DeepFace.analyze(img, actions=['emotion'])
    except:
      print("No face detected")
    
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)    # converting image into smaller size
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)      #Convert BGR to RGB
    
    #Case when multiple faces are identified in webcam
  
    facesCurFrame=face_recognition.face_locations(imgS)   #Find location of the face
    encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)   # TO find encoding of our face in webcam
    
    for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        #display bounding box and name 
        
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            
            y1,x2,y2,x1=faceLoc  #Coordinates of face loc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4 #scaling of image back to original
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)  #Display rectangle of green color of thickness=2 on the image
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)    #Displaying White text of thickness 2 on the frame
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, result['dominant_emotion'],(0,50),font,3,(0,255,0),2,cv2.LINE_4);  
            results=result['dominant_emotion']
            markAttendance(name,results)
            
            
    
    cv2.imshow('webcam',img)     
    
    if cv2.waitKey(1)==13:
        break
        
cap.release()
cv2.destroyAllWindows()
                    