import cv2
import numpy as np
import os

#running video
mp4=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#kNN
def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

def KNN(X, y, k=5):
    dist=[]
    for i in range(X.shape[0]):
        x_train = X[i, :-1]
        y_train = X[i, -1]
        #distance from the test point
        
        d = distance(y, x_train)
        dist.append([d,y_train])
    #sorting based upon the distance
    arr=sorted(dist)
    arr=arr[:k]
    arr=np.array(arr)
    val,cnt=np.unique(arr[:,1], return_counts=True)
    index=cnt.argmax()
    pred=val[index]
    return pred


#preparing training data
class_id=1
mapping={}

face_data=[]
label=[]

face_data_path='./face_data/'

for fx in os.listdir(face_data_path):
    if fx.endswith('.npy'):

        #handling x_train
        data_item = np.load(face_data_path+fx)
        face_data.append(data_item)

        #handling y_train
        target=np.ones(data_item.shape[0])*class_id
        name=fx[:-4]
        mapping[class_id]=name
        label.append(target)
        class_id+=1

face_data=np.concatenate(face_data,axis=0)
label=np.concatenate(label,axis=0).reshape(-1,1)
#whole dataset
dataset=np.concatenate((face_data,label),axis=1)

print(face_data.shape)
print(label.shape)
print(dataset.shape)




#inputing test data
while True:
    ret,frame = mp4.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if(ret==0):
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    #faces = sorted(faces, key = lambda f:f[2]*f[3])

    for face in faces:
        x,y,w,h = face

        offset=10
        face_section = frame[y-offset:y+offset+h, x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(75,75))
        #face_section=np.asarray(face_section)

        #prediction
        pred = KNN(dataset, face_section.flatten())

        #displaying name on the screen
        pred_name = mapping[int(pred)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Video Frame",frame)

    key_pressed = 0xFF & cv2.waitKey(1)
    if key_pressed==ord('q'):
        break




mp4.release()
cv2.destroyAllWindows()

    
    

