import argparse
import time
from pathlib import Path
import os
import pyttsx3 as tts
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
# Command: pip install pillow
from PIL import Image, ImageEnhance

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def train():

  #Initialize names and path to empty list 
  names = []
  path = []

  # Get the names of all the users
  for users in os.listdir("dataset"):
      names.append(users)
      
  # Get the path to all the images
  print(names)
  for name in names:
      for image in os.listdir("dataset/{}".format(name)):
          path_string = os.path.join("dataset/{}".format(name), image)
          path_string = path_string.replace("\\\\","\\")
          path.append(path_string)

  faces = []
  ids = []

  # For each image create a numpy array and add it to faces list
  print(path)
  for img_path in path:
      image = Image.open(img_path).convert("L")

      imgNp = np.array(image, "uint8")
      print(img_path.split("\\")[1].split("_")[0])
      id = int(img_path.split("\\")[1].split("_")[0])

      faces.append(imgNp)
      ids.append(id)

  # Convert the ids to numpy array and add it to ids list
  ids = np.array(ids)

  print("[INFO] Created faces and names Numpy Arrays")
  print("[INFO] Initializing the Classifier")

  # Make sure contrib is installed
  # The command is pip install opencv-contrib-python

  # Call the recognizer
  trainer = cv2.face.LBPHFaceRecognizer_create()
  # Give the faces and ids numpy arrays
  trainer.train(faces, ids)
  # Write the generated model to a yml file
  trainer.write("training.yml")

  print("[INFO] Training Done")

def face_detect(imgg):
  
  cv2.imshow('',imgg)
  img = Image.fromarray(imgg)
  # Load pre-trained face detection model
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

  # Load pre-trained face recognition model
  recognizer = cv2.face.LBPHFaceRecognizer_create()
  train()
  recognizer.read('training.yml')

  # Load list of names corresponding to trained faces
  names = ['Sandesh', 'Praveen', 'Selvin']

  while True:
      # Read frame from webcam
      # ret, img = cap.read()

      # Convert image to grayscale
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # Detect faces in image
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

      # Loop over detected faces
      for (x, y, w, h) in faces:
          # Extract face region from image
          roi_gray = gray[y:y+h, x:x+w]

          # Normalize face region
          roi_gray = cv2.equalizeHist(roi_gray)

          # Recognize face using pre-trained model
          id_, confidence = recognizer.predict(roi_gray)

          # If recognition confidence is high enough, display name
          if confidence < 100:
              name = names[id_]
              cv2.putText(img, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
              print(name,"Detected")
          # Draw rectangle around face
          cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

      # Display image
      cv2.imshow('Face Recognition', imgg)
      

      # Exit on 'q' keypress
      if cv2.waitKey(1) == ord('q'):
          break

  # Release resources
  cap.release()
  cv2.destroyAllWindows()


def detect(source, weights, device, img_size, iou_thres, conf_thres):
    webcam = source.isnumeric()
    
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    isPerson = 0
    imgg = None

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

 
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    t0 = time.perf_counter()

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                # p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im0, frame = path[i], '' , im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if(names[int(c)]) == 'person':
                      imgg = im0
                      # contrast_enhancer = ImageEnhance.Contrast(image)  
                      # pil_enhanced_image = contrast_enhancer.enhance(2)
                      # enhanced_image = np.asarray(pil_enhanced_image)
                      # r, g, b = cv2.split(enhanced_image)
                      # enhanced_image = cv2.merge([b, g, r])
                      face_detect(imgg)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            engine = tts.init()
            engine.say(f'{s}' +'in front of you')
            engine.runAndWait()
            print(f'{s}')

if __name__ == '__main__':
    with torch.no_grad():
        detect("0","yolov7-tiny.pt",device='',img_size=640,iou_thres=0.45, conf_thres=0.25)