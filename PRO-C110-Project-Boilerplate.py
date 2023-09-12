# To Capture Frame
import cv2
import tensorflow as tf
# To process image array 
import numpy as np

# Import the tensorflow modules and load the model
model = tf.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while(True):
      
    status, frame = camera.read()
    img = cv2.resize(frame, (224, 224))
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
    normalise_image = test_image/255.0
    prediction = model.predict(frame)
    print("prediction:", prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    code = cv2.waitKey(1)
    
    if code == 32:
        break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()