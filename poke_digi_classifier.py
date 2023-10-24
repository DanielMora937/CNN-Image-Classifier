
#QUICK NOTE: If you are looking for the answer to 1.3 from the Undergrad Assignment Sheet, check README.md for those answers

import numpy as np #these are all the necessary libraries needed in order to work this classifier
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns #some of these are commented out in the code below, but just are there to show graphical outputs for the dataset
import warnings
import gc
import requests
import tensorflow as tf #We used tensorflow library to be able to produce gradCAM heatmaps later, after the program had run and displayed the classes



from PIL import Image
from io import BytesIO

from lime import lime_image
from collections import Counter #this is again used for datset display, it is all commented out below
from sklearn.model_selection import train_test_split
from skimage.segmentation import mark_boundaries
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D #we decided to use Keras as out model builder, and it is a subclass of TensorFlow
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

warnings.filterwarnings('ignore')

#This is our first block of code, all it does is get the overall data that we will use in building our CNN compiler
path = './data'
classes = os.listdir(path) 
print(f'Total number of categories: {len(classes)}') #This value is equal to 2 as there are two classes within the data set

img_count = {} #We create a dictionary to count up the total amount of data in the classes
for img in classes:
    img_count[img] = len(os.listdir(os.path.join(path, img)))
    
print(f'Total number of images in dataset: {sum(list(img_count.values()))}')

X = [] #List for images
Y = [] #List for labels


for c in classes:
    dir_path = os.path.join(path,c) #Here we make a dataset path for each class
    label = classes.index(c) #This determines the label of the specific class
    
    for i in os.listdir(dir_path):
        img = Image.open(os.path.join(dir_path, i)).convert('RGB') #now we use Pillow to open the images within each class
        try:
            cv_img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) #We convert the Pillow Images into CV2 images as it makes instant numpy arrays for the images
            resized_img = cv.resize(cv_img,(64, 64)) #We resize the image
            X.append(resized_img) #Now the X array that holds all images gets appended with the new CV2 image that has also been resized
            Y.append(label)
        except:
            print(os.path.join(dir_path, i), '[Error] file is unaccessible') #If there is any file that cannot be opened, this except statement will let us know
            continue

print('Import Dataset...Done') #Represents that data has been successfully imported

#temp = Counter(Y)

#fig = plt.figure(figsize = (15,5))
#sns.barplot(x = [classes[i] for i in temp.keys()], y = list(temp.values())).set_title('Number of Images in each Class')
#plt.margins(x=0)
#plt.show()



X = np.array(X).reshape(-1,64,64,3) #here we take the entire array of images, and reshape them to fit our CNN

X = X/255.0 #This scales the images down

y = to_categorical(Y, num_classes = len(classes)) #we convert the format of the labels

#Now we use sklearn to train_test_split the data into a training set and a testing set, which is a 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, shuffle = True, random_state = 42)


#Below is our model, we designed after the VGG-Architecture that is commonly used for image processing in CNN building 
model = Sequential()
#First layer
model.add(Conv2D(16, 3, padding = 'same', activation = 'relu', input_shape = (64, 64, 3), kernel_initializer = 'he_normal')) #This works very similar to using pytorch to create a CNN
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25)) #We use dropout to drop some part of the convolution to reduce overfitting

#Second layer
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2, 2)))

#Third layer
model.add(Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

#Fourth layer
model.add(Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation = 'softmax'))

#model.summary()

#After configuring the VGG-Architecture, and supplying all the portions needed for a functioning model, we compile using the adam optimizer
#and categorical_crossentropy as the loss function to determine the model.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Here we fit the model to history to show how over time the model advances, it will epoch 25 times and based on the graphs, will show proper accuracy and loss values
history = model.fit(x = X_train, y = y_train, batch_size = 32, epochs = 25, validation_data = [X_test, y_test], steps_per_epoch = len(X_train)//32, verbose = 1)

#This is our first figure, we use this to show the models progression overtime
f0 = plt.figure(1)
f0.set_figheight(15)
f0.set_figwidth(20)

plt.subplot(121) #Our first plot is for displaying accuracy
plt.plot(history.history['accuracy'], label = 'acc') #Here, we reference the model's training accuracy 
plt.plot(history.history['val_accuracy'], label = 'val_acc') #This references the model's validation accuracy or test accuracy
plt.legend()
plt.grid()
plt.title(f'accuracy')


plt.subplot(122)#Our second plot is for displaying loss
plt.plot(history.history['loss'], label = 'loss')#This references the training loss
plt.plot(history.history['val_loss'], label = 'val_loss')#This references the test loss
plt.legend()
plt.grid()
plt.title(f'loss')


#Here we create two new arrays, All these images will be used for interpretation results 
pokemon = ['./data/pokemon/130-mega.png', './data/pokemon/321.png', './data/pokemon/419.png', './data/pokemon/607.png']

digimon = ['./data/digimon/120px-Deathmon_black.jpg', './data/digimon/120px-Agumon_hakase.jpg', './data/digimon/120px-Giromon.jpg', './data/digimon/Cardmon_c1.png']

dataset_test = [digimon, pokemon]

val_X = []
val_Y = []

#We cycle through those images and do some conversion to make them numpy arrays and append each as such
for i,imgs in enumerate(dataset_test):
    for img in imgs:
        pic = Image.open(img).convert('RGB')
        cv_pic = cv.cvtColor(np.array(pic), cv.COLOR_RGB2BGR)
        val_X.append(cv_pic)
        val_Y.append(i)

pred_index = None #The values represent our gradCAM heat map and are used to notate inner workings of our models layer
layer_nm = ""
class_channel = None
last_conv_layer_output = None


heatmaps = [] #Since we are working with multiple images, it was easier to store each heatmap in an array to access later
explainers = [] #This array is for our LIME explanations

rows = 9#These variables represent subplots for our other figures
cols = 2

f = plt.figure(2) #This is for graphing our actual model's outputs, while also performing our gradCAM calculations
f.set_figheight(25)#and also our LIME calculations because all images are accessed in this for loop
f.set_figwidth(25)
l_explainer = lime_image.LimeImageExplainer() #We initialize an image explainer that will show contrasts

for i,j in enumerate(zip(val_X, val_Y)):
    true_img = j[0] #Original image at that value j
    true_lbl = j[1] #A 1 or 0 based on what label that item is
    
    pic = cv.resize(true_img, (64,64)) #We perform some image modification to become applicable to our CNN
    pic = pic.reshape(-1, 64, 64, 3)/255.0 #Reshape and scale down the image
    preds = model.predict(pic) #Now we apply that image only to our predictor and it predicts our class
    pred_class = np.argmax(preds) #Here is the value 0 or 1 on which class was predicted
    true = f'True class: {classes[true_lbl]}' #This is used for ploting and showing the difference between actual and predicted
    pred = f'Predicted: {classes[pred_class]} {round(preds[0][pred_class] * 100, 2)}%' #Prediction and percent confidence
    f.add_subplot(rows, cols, i+1)
    plt.imshow(true_img[:, :, ::-1])
    plt.title(f'{true}\n{pred}')
    plt.axis('off')
    f.tight_layout() #This is to be shown in figure 2 as the model took some images from testing and displays its predictions
   
    
    
    #Here we apply a lime explanation to each image after it has undergone reshape and resizing
    lime_exp = l_explainer.explain_instance(pic[0].astype('double'), model.predict, top_labels=3, hide_color = 1, num_samples=1000)
    explainers.append(lime_exp) #We store each explanation in an array to access later in Figure 4 & 5
    
        
    
    #This is the bulk of our gradCAM heatmap
    for layer in reversed(model.layers): #First we find the layer name where the model hits a 4D size, if it doesn't then gradCAM cannot 
        if len(layer.output_shape) == 4: #be applied
            layer_nm = layer.name

    #We implement a grad model that uses our own model
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_nm).output, model.output])
    
    #We do the gradient calculations below
    with tf.GradientTape() as tape:
        last_conv_layer_output, predic = grad_model(pic)
        if pred_index is None:
            pred_index = tf.argmax(predic[0])
        class_channel = predic[:, pred_index] 

    #Here we take the output of the tensorflow gradient tape calculations and create a gradient to use for the heatmaps
    grads = tape.gradient(class_channel, last_conv_layer_output)

    #We pool that gradient
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]

    #Now we create the heatmaps
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    #We will append this array for heat maps to access in Figure 3
    heatmaps.append(heatmap)


#Here we use a similar plotting structure to the one used to show the output of our model
f2 = plt.figure(3)   
f2.set_figheight(15)
f2.set_figwidth(15)
for i,a in enumerate(heatmaps):
    f2.add_subplot(rows, cols, i+1)
    plt.imshow(a) #We show each heat map on a subplot
    plt.title('heatmap')
    plt.axis('off')
    f2.tight_layout()

#Here we take the explainers and use the mask over the image to show the borders that the program is using to determine its classifications
f3 = plt.figure(4)
f3.set_figheight(20)
f3.set_figwidth(20)
for i,a in enumerate(explainers):
    temp, mask = a.get_image_and_mask(a.top_labels[0], positive_only = True, num_features = 5, hide_rest = True)
    f3.add_subplot(rows, cols, i+1)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title('Lime where only positives show')
    plt.axis('off') #It is plotted in a similar manner as previous figures, this shows a different graph in comparison to the next figure

#Same process as above, with slightly modified parameters to show differences
f4 = plt.figure(5)
f4.set_figheight(20)
f4.set_figwidth(20)
for i,a in enumerate(explainers):
    temp_1, mask_1 = a.get_image_and_mask(a.top_labels[0], positive_only= False, num_features = 10, hide_rest = False)  
    f4.add_subplot(rows, cols, i+1)
    plt.imshow(mark_boundaries(temp_1, mask_1))
    plt.title('Lime where everything is shown')
    plt.axis('off')
   
plt.show()

