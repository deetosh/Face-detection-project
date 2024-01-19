import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
facetracker_ = load_model('facetracker.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker_.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [frame.shape[1],frame.shape[0]]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [frame.shape[1],frame.shape[0]]).astype(int)), 
                            (255,255,255), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1],frame.shape[0]]).astype(int), 
                                    [0,-20])),
                      tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1],frame.shape[0]]).astype(int),
                                    [50,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1],frame.shape[0]]).astype(int),
                                               [0,-1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()