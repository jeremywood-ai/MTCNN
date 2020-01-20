# MTCNN
#####
.. image:: https://badge.fury.io/py/mtcnn.svg
    :target: https://badge.fury.io/py/mtcnn

Face detection and face extraction project with MTCNN, Keras, and TensorFlow 2.0, in a Python 3+.

A mutli-tasked cascaded convolutional neural network project shows the ability and speed to detect faces in an image. Additionally, the project's verison 2 allows for image extraction.

.. image:: https://github.com/jeremywood-ai/MTCNN/girl_w_glasses_keypoints.png

This project is based on [Iván de Paz Centeno's MTCNN library](https://github.com/ipazc/mtcnn).

MODEL
#####

"By default the MTCNN bundles a face detection weights model.

"The model is adapted from the Facenet's MTCNN implementation, merged in a single file located inside the folder 'data' relative
to the module's path. It can be overriden by injecting it into the MTCNN() constructor during instantiation.

"The model must be numpy-based containing the 3 main keys "pnet", "rnet" and "onet", having each of them the weights of each of the layers of the network."

*mtcnn_weights.npy* is the weight file from the library that is trained for the three layers.

For more reference about the network definition, take a close look at the paper from *Zhang et al. (2016)* [ZHANG2016]_.

LICENSE
#######

`MIT License`_.


REFERENCE
=========

.. [ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

.. _example.py: example.py
.. _MIT license: LICENSE
