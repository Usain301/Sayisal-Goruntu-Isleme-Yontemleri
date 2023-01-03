# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:50:37 2022

@author: cvkme
"""


import cv2

cap = cv2.VideoCapture("videos/elon_musk.mp4") #cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("datasets/frontalface.xml")


while True:
    ret,frame = cap.read()
    
    if ret == False:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frame = face_cascade.detectMultiScale(gray_frame, 1.3, 3) # x,y,w,h
    
    for (x,y,w,h) in faces_frame:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,148,50), 3)
        
    
    cv2.imshow("Elon Musk",frame)
    
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
    cv2.waitKey(0)
    
cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
