# import Python library
import cv2

# Load trained cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the Given Image
color_image = cv2.imread('Shantonu Acharjee.jpeg')

# Resize the color image
color_image = cv2.resize(color_image, (360, 540))

# convert color image into grayscale image
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Detect Faces (input image, Scasle Factor, Min Neighbors)
faces = face_cascade.detectMultiScale(gray_image, 1.1, 5)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 4)


# Show image
cv2.imshow('Image', color_image)
cv2.waitKey()
cv2.destroyAllWindows()
