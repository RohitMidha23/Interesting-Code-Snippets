import cv2 
import matplotlib.pyplot as plt 
import numpy as np

# variable declarations
lineThickness = 5
x_min = 100
x_max = 1800
y_min = 900
y_max = 1000
interval = 100
image_path = "some_image.jpg"

# read image
image = cv2.imread(image_path)

# draw base lines
cv2.line(image, (x_min, y_min), (x_max, y_min), (0,255,0), lineThickness)
cv2.line(image, (x_min, y_max), (x_max, y_max), (0,255,0), lineThickness)

number = (x_max - x_min)//interval
x = x_min
images = []

# draw vertical lines
for i in range(number+1):
    cropped_image = image[y_min:y_max,x:x+interval]
    cv2.line(image, (x, y_min), (x, y_max), (0,255,0), lineThickness)
    x = x + interval
    images.append(cropped_image)
    plt.imshow(cropped_image)
    plt.show()

# show image
plt.imshow(image)
plt.show()
