import cv2 as cv
import numpy as np

img_orig = cv.imread("Tyre1.jpeg")
# cv.imshow("Input", img_orig)

img = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)

img = cv.GaussianBlur(img, (9,9),0)
img = cv.Canny(img,90,90)
# img = cv.dilate(img, np.ones((2,2),np.uint8), iterations=1)

# cv.imshow("Output", img)

cir = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
cir = np.uint16(np.around(cir))
# img_cir = img_orig
maxr = 0
cirx = 0
ciry = 0
for i in cir[0,:]:
    # cv.circle(img_cir,(i[0],i[1]),i[2],(0,255,0),2)
    # cv.circle(img_cir,(i[0],i[1]),2,(0,0,255),3)
    if ((maxr < i[2]) & (i[2] < img.shape[0]/2)):
        maxr = i[2]
        cirx = i[0]
        ciry = i[1]

# cv.imshow("Output", img_cir)
cv.imshow("Largest", cv.circle(img_orig, (cirx, ciry), maxr, (0,0,255), 2))

print(cirx)
print(ciry)
print(maxr)

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
# cv.imshow("Output", img_crop)

cv.imwrite("CroppedImage.jpg", img_crop)

polar_image = cv.warpPolar(img_orig, dsize=(500,1570) ,center=(cirx, ciry),maxRadius=maxr, flags=cv.WARP_POLAR_LINEAR)
polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imshow("Straightened", polar_image)

img2 = cv.cvtColor(polar_image, cv.COLOR_BGR2GRAY)
img2 = cv.GaussianBlur(img2, (9,9),0)
img2 = cv.Canny(img2,90,90)
cv.imshow("Output", img2)

cv.waitKey()
cv.destroyAllWindows() 