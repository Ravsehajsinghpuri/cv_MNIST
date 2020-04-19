#----------------------------------------IMPORTING NECESSARY PACKAGES AND LIBRARIES----------------------------------#
import tensorflow as tf 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2
import math
from scipy import ndimage

#-------------------------------------------- LOADING THE DATASET AND SPLITTING INTO TRAINING AND TESTING----------------#
mnist=tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
(X_train,X_test)=(X_train/255.0,X_test/255.0)

#-------------------------------------------- BUILDING MODEL ARCHITECTURE------------------------------------------------#

model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
])

predictions=model(X_train).numpy()
tf.nn.softmax(predictions).numpy()

loss_function=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function(Y_train,predictions).numpy()

#------------------------------------------------ COMPILING MODEL AND FITTING PARAMETERS----------------------------------#

model.compile(optimizer='adam',loss=loss_function,metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10)

model.evaluate(X_test,Y_test,verbose=2)

probability_model=tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
])
predictions=probability_model.predict(X_test)
new_predictions=[]
for i in range(len(Y_test)):
  new_predictions.append(np.argmax(predictions[i]))
#------------------------------------------------TESTING OUR MODEL ON TEST IMAGES FROM THE TEST SET------------------------#
test_image_num=1230
plt.imshow(X_test[test_image_num])
print(new_predictions[test_image_num])

#---------------------------------LOADING AND PRE-PROCESSING REAL-LIFE HANDWRITTEN IMAGE FOR OUR MODEL TO PREDICT----------#
def image_preprocess(image_path):
        image_name=image_path
        test_image=cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
        test_image=255-test_image
        test_image=cv2.resize(test_image,(28,28))
        (thresh, test_image) = cv2.threshold(test_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        while np.sum(test_image[0])==0:
          test_image=test_image[1:]

        while np.sum(test_image[:,0])==0:
          test_image=np.delete(test_image,0,1)

        while np.sum(test_image[-1])==0:
          test_image=test_image[:-1]

        while np.sum(test_image[:,-1])==0:
          test_image=np.delete(test_image,-1,1)

        num_rows,num_cols=test_image.shape

        if num_rows>num_cols:
          factor=20.0/num_rows
          num_rows=20
          num_cols=int(round(num_cols*factor))
          test_image=cv2.resize(test_image,(num_cols,num_rows))
        else:
          factor=20.0/num_cols
          num_cols=20
          num_rows=int(round(num_rows*factor))
          test_image=cv2.resize(test_image,(num_cols,num_rows))

        colPadding=(int(math.ceil((28-num_cols)/2.0)),int(math.floor((28-num_cols)/2.0)))
        rowPadding=(int(math.ceil((28-num_rows)/2.0)),int(math.floor((28-num_rows)/2.0)))
        test_image=np.lib.pad(test_image,(rowPadding,colPadding),'constant')

        def calculate_shifts(image):
          center_massx,center_massy=ndimage.measurements.center_of_mass(image)
          num_rows,num_cols=image.shape
          shiftX=np.round(num_cols/2.0-center_massx).astype(int)
          shiftY=np.round(num_rows/2.0-center_massy).astype(int)
          return shiftX,shiftY

        def shift(image,sx,sy):
          num_rows,num_cols=image.shape
          N=np.float32([[1,0,sx],[0,1,sy]])
          shifted_image=cv2.warpAffine(image,N,(num_rows,num_cols))
          return shifted_image

        shiftX,shiftY=calculate_shifts(test_image)
        test_image=shift(test_image,shiftX,shiftY)
        test_image=test_image/255.0
        return test_image
#------------------------SENDING THE PROCESSED TEST IMAGE AS INPUT TO OUR TRAINED MODEL-----------------------------------#

image_path= None    # Place input image path in place of 'None'
test_image=image_preprocess(image_path)
plt.imshow(test_image)
test_image=np.expand_dims(test_image,axis=0)
print(np.argmax(probability_model.predict(test_image)))
