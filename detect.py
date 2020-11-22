import mysql.connector
import cv2
import numpy as np 
import os

mydb = mysql.connector.connect(
  host="localhost",
  user="rozaqku",
  password="rozaqku",
  database="mahasiswa"
)
c = mydb.cursor()

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

face_cascade = cv2.CascadeClassifier('algoFace/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

uname = input("Enter your name: ")

add_user = ("INSERT INTO users "
               "(name) "
               "VALUES (%s)")
data_user = (uname)
c.execute(add_user, data_user)
uid = c.lastrowid
print(c.rowcount, "record inserted.")

sampleNum = 0

while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    sampleNum = sampleNum+1
    cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.waitKey(100)
  cv2.imshow('img',img)
  cv2.waitKey(1);
  if sampleNum > 20:
    break
cap.release()

mydb.commit()
mydb.close()
cv2.destroyAllWindows()