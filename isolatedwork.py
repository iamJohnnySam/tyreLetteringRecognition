import cv2 as cv
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

img_org = cv.imread("1StraightnedImage.jpg")

img = cv.medianBlur(img_org,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()










img = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img, 5)
img = cv.threshold(img, 0 , 255, cv.THRESH_BINARY)

#img = cv.Canny(img,150,150)
#img = cv.dilate(img, np.ones((2,2),np.uint8), iterations=1)
cv.imshow("Output", img)

custom_config = r'--oem 3 --psm 6'
e = pytesseract.image_to_string(img, config=custom_config)
print(e)

cv.waitKey(0)

cv.destroyAllWindows() 