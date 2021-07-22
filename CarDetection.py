import cv2
video=cv2.VideoCapture('carVideo.mp4')
carCaseCade=cv2.CascadeClassifier('cars.xml')
while True:
	src,frame=video.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cars=carCaseCade.detectMultiScale(gray, 1.1, 9)
	for x,y,w,h in cars:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (50,255,255),2)
		cv2.rectangle(frame, (x,y-50),(x+w, y), (50,255,255),-1)
		cv2.putText(frame,"Car",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.75, (255,255,255),2,cv2.LINE_AA)
	frame=cv2.resize(frame, (800,600))
	cv2.imshow("Car Detection",frame)
	wait=cv2.waitKey(1)
	if wait==ord('x'):
		break
video.release()
cv2.destroyAllWindows()
