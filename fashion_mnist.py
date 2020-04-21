import keras
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from scipy import ndimage
import math

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])
train_images=train_images/255.0
test_images=test_images/255.0
train_labels = keras.utils.np_utils.to_categorical(train_labels)
test_labels = keras.utils.np_utils.to_categorical(test_labels)

model=keras.models.Sequential([
      keras.layers.Conv2D(32,(5,5),padding='same',input_shape=(28,28,1)),
      keras.layers.MaxPool2D((2,2)),
      keras.layers.Conv2D(64,(5,5)),
      keras.layers.MaxPool2D((2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(1024,activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=25,verbose=2,batch_size=64)

test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print("\n Accuracy of our model is : {}".format(test_acc))

predictions=model.predict(test_images)
new_predictions=[]
for i in range(len(test_labels)):
  new_predictions.append(np.argmax(predictions[i]))

#test_image_num=3000
#plt.imshow(test_images[test_image_num])
#print(class_names[new_predictions[test_image_num]])

image_name='trouser.jpg'
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
plt.imshow(test_image)
test_image=test_image.reshape(-1,28,28,1)
print(class_names[np.argmax(model.predict(test_image))])
