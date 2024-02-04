import cv2
from random import randrange

#img_file="car.jfif"
img_file="cars.jpg"

classifier_file="car_detector.xml"
car_tracker=cv2.CascadeClassifier(classifier_file)

img=cv2.imread(img_file)

grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates=car_tracker.detectMultiScale(grayscaled_img)
for x,y,w,h in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

cv2.imshow("Car Detector",img)

cv2.waitKey()
print("Code completed")