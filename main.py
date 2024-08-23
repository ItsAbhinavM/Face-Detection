import cv2
import matplotlib.pyplot as plt

faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def imageDetector():
    imagePath='Assets/gandhi.jpeg'
    img=cv2.imread(imagePath)
    grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # face detection
    face = faceClassifier.detectMultiScale(grayImage, scaleFactor=1.8, minNeighbors=0, minSize=(10, 10))

    for (x,y,w,h) in face :
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    # converting BGR image ot RGB
    imageRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,10))
    plt.imshow(imageRGB)
    plt.axis('off')
    plt.show()

def detectBox(vid):
    grayImage=cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    faces=faceClassifier.detectMultiScale(grayImage, scaleFactor=2, minNeighbors=0, minSize=(10, 10))
    for (x,y,w,h) in faces :
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),1)
    return faces

def videoDetector():
    videoCapture=cv2.VideoCapture(0)
    while True:
        result,videoFrame= videoCapture.read()
        if result is False:
            break
        faces=detectBox(videoFrame)
        cv2.imshow("Face detector",videoFrame)
        if cv2.waitKey(1)& 0xFF==ord("q"):
            break
    videoCapture.release()
print("Enter your choice")
print("1: Image facial detection")
print("2: Live facial detection")
choice=int(input("Enter your choice : "))
if choice==1:
    imageDetector()
elif choice==2:
    videoDetector()
else :
    print("Please provide a valid choice")