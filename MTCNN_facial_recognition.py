# -*- coding: utf-8 -*-
"""
@category: computer vision
Facial recognition with MTCNN
"""

# Develop a facial detector with mtcnn

## Step 1: Pulling in the Libraries

from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # Step 4
from matplotlib.patches import Circle # Step 8

## Step 5: Adding Rectangle Definition
def draw_rectangles(filename, result_list):
    # load image
    imageData = plt.imread(filename)
    # plot the image's data
    plt.imshow(imageData)
    # get image axes
    ax = plt.gca() # Get Current Axes (not a typo) method, an object that has two Axis objects (x-axis and y-axis)
    # plot each box
    for result in result_list:
        # coordinates
        x, y, width, height = result['box']
        # The shape
        rectangle = Rectangle((x, y), width, height, fill=False, color='yellow')
        # draw the boxes
        ax.add_patch(rectangle)
        # Step 8
        for key, value in result['keypoints'].items():
            # dot creator
            dot = Circle(value, radius=4, color='red')
            ax.add_patch(dot)
    # Display plot
    plt.show()
## Step 2: set up the model, box sizes and keypoints
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
draw_rectangles(filename, faces) # place the rectanlges on image