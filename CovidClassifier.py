from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape = (224,224,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)
test_dataset = image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r"C:\Users\ACER\Desktop\AI\Covid_Xray\Covid-19-Detection-master\CovidDataset\Train",target_size = (224,224),batch_size = 32,class_mode = 'binary')
validation_set = test_dataset.flow_from_directory(r"C:\Users\ACER\Desktop\AI\Covid_Xray\Covid-19-Detection-master\CovidDataset\Val",target_size = (224,224),batch_size = 32,class_mode = 'binary')

model.fit_generator(training_set,steps_per_epoch = 8,epochs = 10,validation_data = validation_set,validation_steps = 2)

model.save("model.h5")

test_image = image.load_img(r"C:\Users\ACER\Desktop\AI\Covid_Xray\Covid-19-Detection-master\CovidDataset\Val\Normal\NORMAL2-IM-0730-0001.jpeg", target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)
print(result)
print(training_set.class_indices)
