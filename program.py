import cv2 as cv
import numpy as np
import pytesseract

img_orig = cv.imread("Tyre1.jpeg")
imgNu = 3

img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (9,9),0)
img = cv.Canny(img,90,90)

cir = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
cir = np.uint16(np.around(cir))
maxr = 0
cirx = 0
ciry = 0
for i in cir[0,:]:
    if ((maxr < i[2]) & (i[2] < img.shape[0]/2)):
        maxr = i[2]
        cirx = i[0]
        ciry = i[1]

img_lar = cv.circle(img_orig, (cirx, ciry), maxr, (0,0,255), 2)
# cv.imshow("Largest", img_lar)
cv.imwrite(str(imgNu)+"DetectedImage.jpg", img_lar)

print(cirx)
print(ciry)
print(maxr)

x = np.array([cirx, ciry])

x1 = 0
x2 = img.shape[1]-1
y1 = 0
y2 = img.shape[0]-1

if(ciry>maxr):
    y1 = ciry-maxr

if(ciry+maxr<img.shape[0]-1):
    y2 = ciry+maxr

if (cirx>maxr):
    x1 = cirx-maxr

if (cirx+maxr<img.shape[1]-1):
    x2 = cirx+maxr

img_crop = img_orig[y1:ciry+maxr, x1:cirx+maxr]
#cv.imshow("Output", img_crop)

cv.imwrite("CroppedImage.jpg", img_crop)

img_crop1 = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
img_crop1 = cv.GaussianBlur(img_crop1, (9,9),0)
img_crop1 = cv.Canny(img_crop1,90,90)

ccir = cv.HoughCircles(img_crop1,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
ccir = np.uint16(np.around(cir))

center = np.empty(shape=[0, 2])

img_crop3 = img_crop

for i in cir[0,:]:
    if ((i[0] > cirx*0.9) & (i[0] < cirx*1.1) & (i[1] > ciry*0.9) & (i[1] < ciry*1.1)):
    # if ((i[2] > img_crop.shape[0]*0.3) & (i[2] < img_crop.shape[0]*0.55) & (i[0] > cirx*0.9) & (i[0] < cirx*1.1) & (i[1] > ciry*0.9) & (i[1] < ciry*1.1)):
        center = np.append(center, [[i[0], i[1]]], axis=0)
        #img_crop3 = cv.circle(img_crop3, (i[0], i[1]), i[2], (0,0,255), 2)

# cv.imshow("circles", img_crop3)

# x = np.mean(center, axis=0)

print (x)

polar_image = cv.warpPolar(img_orig, dsize=(500,1700) ,center=(x[0], x[1]),maxRadius=maxr, flags=cv.WARP_POLAR_LINEAR)
polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imshow("Straightened", polar_image)
cv.imwrite(str(imgNu)+"StraightnedImage.jpg", polar_image)

img2 = cv.cvtColor(polar_image, cv.COLOR_BGR2GRAY)
img2 = cv.GaussianBlur(img2, (5,5),0)
img2 = cv.Canny(img2,90,90)
# cv.imshow("Output", img2)
cv.imwrite(str(imgNu)+"RecognitionImage.jpg", img2)

from pytesseract import Output
custom_config = r'--oem 3 --psm 6'
d= pytesseract.image_to_string(img2, config=custom_config)
# d = pytesseract.image_to_data(img2, output_type=Output.DICT)
print(d)


cv.waitKey(0)
cv.destroyAllWindows() 