import numpy as np 
import os 
from matplotlib import pyplot as plt
from train import *
from random import randrange


test_imgs_dir= './test/'


def get_classes():
    classes = training_data.class_indices
    classes = {v: k for k, v in classes.items()}

    return classes


def classify_batch(batch = 20):

    classes = get_classes()

    img_list = os.listdir(test_imgs_dir)
    images_number = len(img_list)

    index = randrange(0,images_number,1)

    batch_holder = np.zeros((batch, image_height, image_width, 3))
    for i,img in enumerate(img_list[index: index + batch]):
        img = tf.keras.preprocessing.image.load_img(os.path.join(test_imgs_dir,img), target_size=(image_height,image_height))
        batch_holder[i, :] = img


    result = model.predict_classes(batch_holder)
    
    fig = plt.figure(figsize=(25, 25))
    
    for i,img in enumerate(batch_holder):
        fig.add_subplot(4,5, i+1)
        plt.title(classes[result[i]])
        plt.axis('off')
        plt.imshow(img/256.) 

    plt.show()



def classify_image(image):
    classes = get_classes()

    img = tf.keras.preprocessing.image.load_img(image, target_size=(image_height,image_width))
    img = np.array(img)
    img = img.reshape(1, image_height,image_width, 3)
    pred = np.argmax(model.predict(img))

    print(classes[pred])
