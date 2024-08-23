import cv2
import matplotlib.pyplot as plt

imagePath='Assets/gandhi.jpeg'
img=cv2.imread(imagePath)
grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# face detection
face = faceClassifier.detectMultiScale(grayImage, scaleFactor=1.8, minNeighbors=0, minSize=(10, 10))

print(face)
for (x,y,w,h) in face :
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

# converting BGR image ot RGB
imageRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.imshow(imageRGB)
plt.axis('off')
plt.show()
