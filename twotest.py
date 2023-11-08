# Import modules
import torch
import torchvision
import numpy as np
from PIL import Image
# from torchvision.models import swin_transformer
import time
from glob import glob
from tqdm import tqdm
import pickle
import cv2
import pandas
import os
import shelve
from model import ft_net
from utils2 import fuse_all_conv_bn


model_structure = ft_net(751, stride = 2, ibn = False, linear_num=512)

def load_network(network):
    save_path = 'model/ft_ResNet50/net_last.pth'
    network.load_state_dict(torch.load(save_path))
    return network

model = load_network(model_structure)
model = model.eval()
# model = model.cuda()
model = fuse_all_conv_bn(model)

# Load models
# model = torchvision.models.swin_transformer.swin_s(pretrained=True)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# Define functions
def save_features(feature, file, id):
    # Store the feature vector under the given id
    with shelve.open(file, 'c') as shelf: # Use 'c' flag to create the file if it does not exist or open it for appending if it does
        shelf[str(id)] = feature # Convert the id to a string and use it as the key
    
def load_features(file, id):
    # Retrieve the feature vector from the given id
    with shelve.open(file, 'r') as shelf: # Use 'r' flag to open the file for reading
        feature = shelf[str(id)] # Convert the id to a string and use it as the key
    return feature

    
def load_image(path):
    # Load the image from the given path
    image = Image.open(path)
    # Resize and center crop the image to 224x224 pixels
    image = torchvision.transforms.Resize(256)(image)
    image = torchvision.transforms.CenterCrop(224)(image)
    # Convert the image to a tensor and normalize it
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )(image)
    # Add a batch dimension to the image
    image = image.unsqueeze(0)
    return image

def extract_features(image):
    # Pass the image through the model and get the output of the last layer
    output = model(image)
    # Flatten the output and return it as a feature vector
    feature = output.view(-1)
    return feature

def euclidean_distance(feature1, feature2):
    # Compute the squared difference between the two vectors
    diff = feature1 - feature2
    diff = diff ** 2
    # Sum up the squared differences and take the square root
    distance = torch.sum(diff)
    distance = torch.sqrt(distance)
    return distance

# Define parameters
# source = 'output2.mp4' # camera index or video file path
source = 'rtsp://admin:Admin%4012345@192.168.241.57:554/1/1'
threshold = 10 # distance threshold for assigning IDs

# Define variable for tracking current ID
current_id = 0

# Define variable for storing video capture object
cap = cv2.VideoCapture(source)

# Define variable for storing video writer object
out = cv2.VideoWriter(f'output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))

# Define variable for storing timer
timer = 0

# Define variable for storing flag
need_feature = True

# Define variable for storing start time of loop
start_time = time.time()

# Define variable for storing count of frames
count = 0

try:
    # Start loop
    while True:
        # Read frame from video capture object
        ret, frame = cap.read()

        # Check if frame is valid
        if not ret:
            break

        # Get current time
        current_time = time.time()

        # Check if 2 seconds have passed since the last feature extraction
        if current_time - timer > 2:
            # Set flag to True
            need_feature = True
            # Reset timer
            timer = current_time

        # Pass frame through YOLOv5 model and get predictions
        predictions = yolo_model(frame)
        df = predictions.pandas().xyxy[0]

        

        # Loop over the rows of the DataFrame
        for index, row in df.iterrows():
            if row['class'] == 0: # If class is person
                xmin = int(row['xmin'])
                xmax = int(row['xmax'])
                ymin = int(row['ymin'])
                ymax = int(row['ymax'])

                # Draw bounding box on frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)

                # Crop frame using coordinates of bounding box
                cropped_frame = frame[ymin:ymax, xmin:xmax]

                # Check if flag is True
                if need_feature:
                    # Convert cropped frame to PIL image and tensor
                    cropped_pil_image = Image.fromarray(cropped_frame)
                    cropped_tensor_image = torchvision.transforms.ToTensor()(cropped_pil_image).unsqueeze(0)

                    # Extract features from cropped tensor image using Swin Transformer
                    feature = extract_features(cropped_tensor_image)

                    # Set flag to False
                    need_feature = False

                # Assign a temporary ID to the bounding box
                temp_id = current_id

                # Increment current ID
                current_id += 1

                # Define a flag for indicating if a match is found
                match_found = False

                # Loop through the ids in the shelf file and compare features
                with shelve.open("features.shelf", 'c') as shelf: # Open the file for reading
                    for boxid in shelf.keys(): # Loop over the keys
                        # Load the feature vector from the file
                        feat = load_features("features.shelf", boxid)
                        # Calculate Euclidean distance between features
                        distance = euclidean_distance(feature, feat)

                        # If distance is less than threshold, assign same ID as key
                        if distance < threshold:
                            temp_id = int(boxid) # Convert the key to an integer
                            match_found = True
                            break

                # If no match is found, save new feature and ID to the file
                if not match_found:
                    save_features(feature, "features.shelf", temp_id)

                # Draw ID on frame
                cv2.putText(frame, f"ID: {temp_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {temp_id}", (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


        # Display frame with annotations
        cv2.imshow("Frame", frame)
        out.write(frame)

        # Save bbox_features dict to file
        # save_features(bbox_features, "features.shelf")

        # Reset count to zero
        count = 0

        # Wait for key press and exit if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Print elapsed time of loop
        print(f"Elapsed time: {time.time() - start_time} seconds")
except Exception as e:
    print(e)

# Release video capture object and video writer object and destroy windows
cap.release()
out.release()
cv2.destroyAllWindows()

