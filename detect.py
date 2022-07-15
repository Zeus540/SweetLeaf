"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from picamera.array import PiRGBArray
from picamera import PiCamera
import datetime
import numpy as np
import requests
import base64
import json
import my_file 

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """
  date = datetime.date.today()
  t = time.localtime()
  current_time = time.strftime("%H:%M:%S", t)
  dposition =  (10, 30) 
  tposition =  (10, 60) 

  # Start capturing video input from the camera
  camera = PiCamera()
  #camera.shutter_speed = 10000
  camera.resolution = (1920,1088)
  #camera.awb_mode = 'off'
  #camera.awb_gains=(0.4,1)
  camera.rotation = 0
  camera.zoom = (0.1, -0.5, 0.9, 0.8)
 # camera.saturation = 2
  camera.brightness = 43
  #camera.contrast = 2
  time.sleep(2)

  imageCapture = PiRGBArray(camera)
  camera.capture(imageCapture, format="bgr" )
  image = imageCapture.array
  camera.close()
  flipped = cv2.flip(image,0)
  npArray = np.array(image) 

  

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name="./main.tflite", use_coral=enable_edgetpu, num_threads=2)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.5)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)


  # Convert the image from BGR to RGB as required by the TFLite model.
  rgb_image = cv2.cvtColor(npArray, cv2.COLOR_BGR2RGB)

  # Create a TensorImage object from the RGB image.
  input_tensor = vision.TensorImage.create_from_array(rgb_image)

  # Run object detection estimation using the model.
  detection_result = detector.detect(input_tensor)


  # Draw keypoints and edges on input image
  image = utils.visualize(npArray, detection_result)

  cv2.imwrite("images/"+str(date)+current_time+'.png',npArray) 

  #readings = my_file.get_data()
  readings = my_file.get_data()

#  print(readings)
  image_file = "images/"+str(date)+current_time+'.png'
  with open(image_file,"rb") as f:
      im_bytes = f.read()

  im_bb64 = base64.b64encode(im_bytes).decode("utf8")
  files = json.dumps({"image": im_bb64,"name":str(date)+current_time,"readings":readings})
  url = "http://api.odinsgate.co.za/nodemcu"
  headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
  test_response = requests.post(url, data=files,headers=headers)

  print(test_response.json)
  


  # cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
