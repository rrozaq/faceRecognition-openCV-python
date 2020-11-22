import mysql.connector
import cv2
import numpy as np 
import sqlite3
import os

mydb = mysql.connector.connect(
  host="localhost",
  user="rozaqku",
  password="rozaqku",
  database="mahasiswa"
)
c = mydb.cursor()

fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

face_cascade = cv2.CascadeClassifier('algoFace/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
		ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
		select = ("select name from users "
               "WHERE id = (%s)")
		where = (ids,)
		c.execute(select, where)
		
		# c.execute("select name from users where id = (?);", (ids,))
		result = c.fetchall()
		print(result)
		name = result[0][0]
		if conf < 50:
			cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
		else:
			cv2.putText(img, 'Tidak Tahu', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
	cv2.imshow('Face Recognizer',img)
	if cv2.waitKey(1) & 0xff == ord('x'):
		break

cap.release()
cv2.destroyAllWindows()