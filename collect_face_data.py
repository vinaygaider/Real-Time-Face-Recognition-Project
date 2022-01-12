import cv2
import numpy as np
import os, sys

#0=id of 1st camera, mp4=running video
mp4=cv2.VideoCapture(0)

#importing the haarcascade_classifier
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count=0

#storing the face data
face_data=[]
face_data_path='./face_data/'
file_name=input("Enter the name of the person : ")


while True:
    
    #frame=a single image of running video, ret=return value
    ret,frame=mp4.read()                                                
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:                                                      
        continue

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    #if there are multiple faces, we are going to select the largest face of all
    faces=sorted(faces, key=lambda f:f[2]*f[3])
    
    face_section=0
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) #gray_frame<->frame

        #cropping to get the region of interest
        offset = 10
        
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset] #gray_frame<->frame
        face_section = cv2.resize(face_section,(75,75))
        

        #taking the 10th image
        count += 1
        if(count%10==0):
            face_data.append(face_section)
            print(len(face_data))

    
    
    cv2.imshow("Video Frame",frame)
    cv2.imshow("Face",face_section)

    key_pressed=cv2.waitKey(1)& 0xFF
    if key_pressed==ord('q') or len(face_data)==10:
        break

#converting list into array without creating a copy
face_data1 = np.asarray(face_data)
#transfroming horizonatal to vertical
face_data1=np.reshape(face_data1,(face_data1.shape[0],-1))
print(face_data1.shape)

#saving the data into the system
np.save(face_data_path+file_name+'.npy',face_data1)

print("Data successfully saved at "+face_data_path+file_name+'.npy')



mp4.release()
cv2.destroyAllWindows()
