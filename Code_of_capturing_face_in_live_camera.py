import cv2

mp4=cv2.VideoCapture(0)                                                 #0==id of 1st camera, mp4==running video
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret,frame=mp4.read()                                                #frame==a single image of running video
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:                                                      #ret=return value
        continue

    
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    print(faces)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
    cv2.imshow("Video Frame",frame)
    
    key_pressed=cv2.waitKey(1)& 0xFF
    if key_pressed==ord('q'):
        break

mp4.release()
cv2.destroyAllWindows()
