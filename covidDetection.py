# Covid Detection using CNN
# COVID-19 DETECTION FROM X-RAY
# -> Blood tests are costly
# -> Blood tests take time to conduct ~5 hours per patient
# ->Extent of spread can be detected
# ->Build deep learning models
# ->can be classified using image classification models & segmentation techniques
# -> Dataset available

import os
train_path = "Data/train"
val_path = "Data/validation"
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# MODEL
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss=keras.losses.binary_crossentropy,optimizer="adam",metrics=["accuracy"])
model.summary()

# TRain it from scratch
train_datagen = image.ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = image.ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory("D:\Aiml\Data Science CB\\63\Data\\train",batch_size=32,target_size=(224,224),class_mode='binary')
val_generator = val_datagen.flow_from_directory("D:\Aiml\Data Science CB\\63\Data\\validation",batch_size=32,target_size=(224,224),class_mode='binary')
print(train_generator.class_indices)

hist = model.fit(train_generator,epochs=2,steps_per_epoch=8,validation_data=val_generator,validation_steps=4)

y_test = []
y_actual = np.array([np.zeros(15),np.ones(15)])

from keras.utils import load_img,img_to_array
for i in os.listdir("D:\Aiml\Data Science CB\\63\Data\\test\\"):
    img = load_img("D:\Aiml\Data Science CB\\63\Data\\test\\"+i,target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    p = model.predict(img)
    if(p[0,0]>0.5):
        y_test.append(1)
    else:
        y_test.append(0)

print(y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_actual,y_test)
import seaborn as sns
sns.heatmap(cm,cmap='plasma',annot=True)
plt.show()