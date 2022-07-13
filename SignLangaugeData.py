from scripts.extractframes import Vid2Frame

# vid = 'videoplayback.mp4'
# path = 'D:\\projects\\signlangauge\\signlangaugecodes\\VideoData'
# Vid2Frame(vid,path = path)



import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('D:\\projects\\signlangauge\\signlangaugecodes\\VideoData\\000056.jpg')

new_img  = np.zeros((1024,1024,3))

x_mid,y_mid = img.shape[0] / 2.0, img.shape[1] / 2.0
new_x_mid, new_y_mid =  new_img.shape[0] / 2.0, new_img.shape[1] / 2.0

print(new_x_mid,new_y_mid)

plt.imshow(img[:, 100:330])
plt.show()