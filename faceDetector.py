import cv2

# using Open CV3 Cascade Classifier to read the face detector cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# reading the image to examine
img = cv2.imread("My_phot.jpg")
# converting the image into a 2 dminsion balck and white image
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#detecting the faces within the image
faces= face_cascade.detectMultiScale(gray , scaleFactor = 1.05 , minNeighbors = 5)

#looping through faces and drawing a rectangle arround each face
for x , y , w, h in faces:
    cv2.rectangle(img , (x,y) , (x+w , y+h) , (0, 255,0), 1)
    
   #showing the image with rectangles drawn in it 
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
