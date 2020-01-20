# -*- coding: utf-8 -*-
"""
@category: computer vision
Facial Extraction with MTCNN (Version 2)
"""

# Develop a facial extraction with mtcnn

from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

## Step 5: Adding Rectangle Definition
def find_faces(filename, result_list):
    # read the image for detection
    imageData = plt.imread(filename)
    # find the face bounding areas (subplots)
    for c in range(len(result_list)):
        # find coordinates of the box 'c'
        x1, y1, width, height = result_list[c]['box']
        # find the other side of the box
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(result_list), c+1)
        plt.axis('off')
        # disply the face
        plt.imshow(imageData[y1:y2, x1:x2])
    plt.show()

# setup the model variable that calls for a weights file
model = MTCNN(weights_file='mtcnn_weights.npy')

## Step 3: Load image
filename = 'group2013.jpg'
pixel_array = plt.imread(filename) # Read an image into an array
# establish detector with default weights
detector = model
# find facial features in image
faces = detector.detect_faces(pixel_array)
for face in faces:
    print(face) # print out the coordinates (optional)
## Step 6: Add the call to the Draw Function
find_faces(filename, faces) # place the rectangles on image