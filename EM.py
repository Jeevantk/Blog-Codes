import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread("face.jpeg",0) #Read the Image as a gray scale image
kernel=np.ones((5,5),np.float32)/25 # define the kernel

blur_img=cv2.filter2D(img,-1,kernel) # Convolving with the kernel

plt.hist(blur_img.ravel(),256,[0,256]); plt.show() # plot the histogram

# hist = cv2.calcHist([blur_img],[0],None,[256],[0,256]) # calculating the histogram

points=[]
for i in range(blur_img.shape[0]):
	for j in range(blur_img.shape[1]):
		points.append(img[i][j])

points = np.float32( np.vstack(points) )

cluster_no=3
em=cv2.ml.EM_create()
em.setClustersNumber(cluster_no)
em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
# em.trainEM(points)
# means = em.getMeans()
# covs = em.getCovs()

# print means
# print covs

ret,thresh = cv2.threshold(img,178,255,cv2.THRESH_BINARY)

cv2.imshow("Skin Color",thresh)
cv2.imwrite("thresh.jpg",thresh)
cv2.waitKey(0)