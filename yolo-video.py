from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os




#res=youtube.videos().list(part='snippet,contentDetails,statistics',chart='mostPopular',regionCode='IN',videoCategoryId='',maxResults=15).execute() 


#clearing the folder before capturing video frames.

##net = cv2.dnn.readNetFromCaffe(prototxt,model)

# initialize the video stream and allow the cammera sensor to warmup
#print("[INFO] starting video stream...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()


#target=cv2.dnn.DNN_TARGET_OPENCL_FP16

yolo="C:\\Users\\shivu\\Downloads\\python notes\\project-yolo\\we-yolo"
objects={"lays":0,"kurkure":0,"apple":0,"coke":0,"colgate":0}
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo, "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")
# derive the paths to the YOLO weights and model configuration

weightsPath = os.path.sep.join([yolo, "my-yolo2.backup"])
configPath = os.path.sep.join([yolo, "my-yolo2.cfg"])
 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# load our input image and grab its spatial dimensions

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities


# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

##labelsPath = os.path.sep.join([yolo, "coco.names"])
##LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
##weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
##configPath = os.path.sep.join([yolo, "yolov3.cfg"])
(W, H) = (None, None)

#net.setPreferableTarget(target)
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frame = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
   
    
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        objects={"lays":0,"kurkure":0,"apple":0,"coke":0,"colgate":0}
        # loop over each of the detections
        for detection in output:
            
            
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
        
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.0:
                
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            objects[LABELS[classIDs[i]]]+=1
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    cv2.imshow("Frame", frame)
    elap = (end - start)
    if any(objects.values()):
        print(objects)
    key=cv2.waitKey(3) & 0xFF
    fps.update()
        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        WebcamVideoStream(src=0).stop()
        vs.stream.release()
        break
    
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))     
vs.stop()
cv2.destroyAllWindows()    

# release the file pointers
print("[INFO] cleaning up...")





