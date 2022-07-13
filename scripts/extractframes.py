import cv2
import os
import sys
from glob import glob



def Vid2Frame(custom_video, path = "/content/imagedata/"):
    count=1
    vidcap = cv2.VideoCapture(custom_video)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            # dim = (512, 512) # you can change image height and image width
            # resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            print('The img count is', count)
            print('Written at', os.path.join(path,str(count).zfill(6))+".jpg")
            cv2.imwrite(os.path.join(path,str(count).zfill(6))+".jpg", image)#, image[0:150, 75:250]) # image write to image folder be sure crete image folder in same dir
        return hasFrames

    sec = 0
    frameRate = 0.1 # change frame rate as you wish, ex : 30 fps => 1/30

    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)



def ConvertFormat(folderpath, fomat = '.jpg'):
    imgs = os.listdir(folderpath)
    for i in imgs:
        name = i.split('.')
        new_name = name[0]+format
        os.rename(os.path.join(folderpath,i), os.path.join(folderpath,new_name))



def resize_images(folderpath, img_shape):
    width, height, depth = img_shape
    for imgs in os.listdir(folderpath):
        img = cv2.imread(os.path.join(folderpath, imgs))
        img = cv2.resize(img,(width,height))
        cv2.imwrite(os.path.join(folderpath, imgs), img)



