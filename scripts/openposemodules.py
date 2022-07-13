# Import general libraries
import sys
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import numpy as np



# from google.colab.patches import cv2_imshow
# # Set Python Openpose Directory for python api (Important)
# pyopenpose_dir = os.path.join(OpenposeDir,'build','python') # ex: '/content/openpose/build/python'
# if pyopenpose_dir not in sys.path:
#     sys.path.append(pyopenpose_dir)



def GetOpenPoseInfo(img_path,show = True, OpenposeDir = '/content/openpose/'):

  # Custom Params (refer to openpose/include/openpose/flags.hpp for more parameters)
  from openpose import pyopenpose as op



  params = dict()
  params["model_folder"] = os.path.join(OpenposeDir,'models')  # ex: '/content/openpose/models'
  params["face"] = True
  params["hand"] = True

  # Starting OpenPose
  opWrapper = op.WrapperPython()
  opWrapper.configure(params)
  opWrapper.start()

  # Process Image
  datum = op.Datum()
  input_image = cv2.imread(img_path) # Change Image Here
  datum.cvInputData = input_image
  opWrapper.emplaceAndPop(op.VectorDatum([datum]))
  network_output = datum.poseKeypoints

  # Display Image
  if show  ==  True:
    cv2_imshow(datum.cvOutputData)

  #Information about points
  # print("Body keypoints: \n" + str(datum.poseKeypoints))
  # print("Face keypoints: \n" + str(datum.faceKeypoints))
  # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
  # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))


  return (datum.poseKeypoints, datum.faceKeypoints, datum.handKeypoints[0], datum.handKeypoints[1])


def refineposedata(data):
  refineddata = []
  for no,point in enumerate(data):
    if point[0] == 0 and point[1] == 0:
      refineddata.append(None)
    else:
      refineddata.append((point[0], point[1]))
  return refineddata




def DrawSkeletonFrames(imgpath, skeletonpath):

    
  pose, face, lefthand, righthand = GetOpenPoseInfo(imgpath, show = False)


  POSE_PAIRS = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   
              [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0],   [0,15], [15,17],  
              [0,16], [16,18],   [14,19],[19,20],  [14,21], [11,22],
              [22,23],[11,24]]

  FACE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], 
                [17, 18], [18, 19], [19, 20], [20, 21],
                [22, 23], [23, 24], [24, 25], [25, 26],
                [27, 28], [28, 29], [29, 30],
                [31, 32], [32, 33], [33, 34], [34, 35],
                [36, 37], [37, 38], [38, 39], [39, 40], [40, 41],
                [42, 43], [43, 44], [44, 45], [45, 46], [46, 47],
                [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59],[48,59],
                [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67],[60,67],[60,61]
                ]

  HANDS_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], 
                  [0, 5], [5, 6], [6, 7], [7, 8], 
                  [0, 9], [9, 10], [10, 11], [11, 12], 
                  [0, 13], [13, 14], [14, 15], [15, 16], 
                  [0, 17], [17, 18], [18, 19], [19, 20]]

  nPoints = 25


  frame = cv2.imread(imgpath) 
  frame = np.zeros(frame.shape)

  points = refineposedata(pose[:,:,:2][0])
  # Draw Skeleton
  for pair in POSE_PAIRS:
      partA = pair[0]
      partB = pair[1]
      
      if points[partA] and points[partB]:
          cv2.line(frame, points[partA], points[partB], (0, 0, 255), 3)

  points = refineposedata(face[:,:,:2][0])
  # Draw Skeleton
  for pair in FACE_PAIRS:
      partA = pair[0]
      partB = pair[1]
      
      if points[partA] and points[partB]:
          cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)


  points = refineposedata(lefthand[:,:,:2][0])
  # Draw Skeleton
  for pair in HANDS_PAIRS:
      partA = pair[0]
      partB = pair[1]
      
      if points[partA] and points[partB]:
          cv2.line(frame, points[partA], points[partB], (255, 0,0), 3)


  points = refineposedata(righthand[:,:,:2][0])
  # Draw Skeleton
  for pair in HANDS_PAIRS:
      partA = pair[0]
      partB = pair[1]
      
      if points[partA] and points[partB]:
          cv2.line(frame, points[partA], points[partB], (225, 0,0), 3)


  cv2.imwrite(skeletonpath, frame)