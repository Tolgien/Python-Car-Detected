import cv2
from random import randrange

#img_file="car.jfif"
#img_file="cars.jpg"
video=cv2.VideoCapture("2.mp4")

classifier_file="car_detector.xml"
car_tracker=cv2.CascadeClassifier(classifier_file)

pedestrian_tracker=cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    #successful_fram_read = True the only frame will operate
    successful_frame_read,frame=video.read()

    grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect cars and people
    cars=car_tracker.detectMultiScale(grayscaled_frame,1.1)
    pedestrian=pedestrian_tracker.detectMultiScale(grayscaled_frame,1.1,1)

    for x,y,w,h in pedestrian:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    for x,y,w,h in cars:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)

    cv2.imshow("Car Detector",frame)

    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()
print("Code completed")