# OpenCV_projects

### List of computer vision projects using OpenCV library.
##
- Object_Measurement:

  This project measure, in real time, contours of objects in a video with an aruco marker (at the same horizontal plane) through the following steps:
  - 1 - Aplication of an edge detection mask.
  - 2 - Extraction of object contours and bounding boxes with width and height.
  - 3 - Obtaining ratio of pixels per centimeters, and therefore, object mesurements, using the known dimensions of the Aruco Marker (5x5cm).
  
  Example image of a frame after steps:
  ![image](https://github.com/Yuri-Vlasqz/OpenCV_projects/blob/1b53f8b86c5175ea884dbb27b46204c184b898f8/Object_Measurement/test%20image%20GaussianBlur%20Canny.jpg)
  - Objects bounding boxes in blue
  - Objects mesurementes in dark green 
  - Main Aruco Marker in light green
