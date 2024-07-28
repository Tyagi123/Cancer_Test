#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# In[ ]:


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image

# In[3]:


import pandas as pd
import numpy as np
from pathlib import Path

# In[5]:


TEST_IMG_DIR = "/Users/gauravkumar/Downloads/Data/isic-2024-challenge/train-image"
TEST_CSV_DIR = "/Users/gauravkumar/Downloads/Data/isic-2024-challenge/train-metadata.csv"

# In[7]:


csv_data = pd.read_csv(TEST_CSV_DIR, low_memory=False);

# In[8]:


csv_data.columns

# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[10]:


train_datagen = ImageDataGenerator(
    rescale=1. / 512,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# In[11]:


train_generator = train_datagen.flow_from_directory(
    TEST_IMG_DIR,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary'
)

# In[12]:


train_generator.image_shape

# In[19]:


import matplotlib.pyplot as plt

# In[17]:


from tensorflow.keras import datasets, layers, models


# In[21]:


def getCropImgs(img, needRotations=False):
    # img = img.convert('L')
    z = np.asarray(img, dtype=np.int8)
    c = []
    for i in range(3):
        for j in range(4):
            crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]

            c.append(crop)
            if needRotations:
                c.append(np.rot90(np.rot90(crop)))

    # os.system('cls')
    # print("Crop imgs", c[2].shape)

    return c


# In[23]:


from sklearn.model_selection import train_test_split

# In[39]:


data_new = csv_data[['isic_id', 'target']]

# In[40]:


data_new.columns

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(csv_data, csv_data['target'], test_size=0.33, random_state=4,
                                                    stratify=csv_data['target'])

# In[26]:


x_train.columns

# In[31]:


y_train.size

# In[32]:


x_train.size

# In[35]:


modelSavePath = '/Users/gauravkumar/Downloads/my_model3.h5'


# In[36]:


def defModel(input_shape):
    X_input = Input(input_shape)

    # The max pooling layers use a stride equal to the pooling size

    X = Conv2D(16, (3, 3), strides=(1, 1))(X_input)  # 'Conv.Layer(1)'

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(2)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Conv.Layer(3)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(4)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(5)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(5) will be 82x82, we want 84x84

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(6)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(7)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(7) will be 40x40, we want 42x42

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(8)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Con.Layer(9)

    X = Activation('relu')(X)

    X = Flatten()(X)  # Convert it to FC

    X = Dense(256, activation='relu')(X)  # F.C. layer(10)

    X = Dense(128, activation='relu')(X)  # F.C. layer(11)

    X = Dense(4, activation='softmax')(X)

    # ------------------------------------------------------------------------------

    model = Model(inputs=X_input, outputs=X, name='Model')

    return model


# In[37]:


def train(batch_size, epochs):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    model = defModel(X_train.shape[1:])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # Uncomment the below code and comment the lines with(<>), to implement the image augmentations.

    # datagen = keras.preprocessing.image.ImageDataGenerator(
    # zoom_range=0.2, # randomly zoom into images
    # rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    # horizontal_flip=False,  # randomly flip images
    # vertical_flip=False  # randomly flip images
    # )
    while True:
        try:
            model = load_model(modelSavePath)
        except:
            print("Training a new model")

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)  # <>

        # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
        #                              epochs=epochs
        #                              # validation_data=(X_test, Y_test))
        #                              )
        # history.model.save('my_model3.h5')

        model.save(modelSavePath)

        preds = model.evaluate(X_test, y_test, batch_size=1, verbose=1, sample_weight=None)
        print(preds)

        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")
        ch = input("Do you wish to continue training? (y/n) ")
        if ch == 'y':
            epochs = int(input("How many epochs this time? : "))
            continue
        else:
            break

    return model


# In[38]:


# Get the softmax from folder name
def getAsSoftmax(fname):
    if (fname == 'b'):
        return [1, 0]
    else:
        return [0, 1]


# In[41]:


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


# In[ ]:


import os


# In[ ]:


def addImageDataToPd(basePath, data_new):
    print("Hello---")
    data_new['img'] = data_new.apply(lambda _: '', axis=1)
    data_new['imgcrop'] = data_new.apply(lambda _: '', axis=1)
    print("Hello1---")
    # for i in data_new['isic_id']:
    for i in range(4):
        print("isic_id= " + data_new['isic_id'])
        img = Image.open(os.path.join(os.path.join(basePath, "/"), data_new['isic_id'] + ".jpg"))
        data_new['imgcrop'] = getCropImgs(img)
        data_new['img'] = img


# In[ ]:


addImageDataToPd(TEST_IMG_DIR, data_new)

# In[ ]:




