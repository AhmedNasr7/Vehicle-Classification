import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model


image_height = 300
image_width = 300

batch_size = 32

def import_data(train_dir, batch_size = batch_size):

    
    
    train_generator = ImageDataGenerator(rescale = 1. / 255, 
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode = 'nearest',
                                        validation_split=0.2)

    train_data = train_generator.flow_from_directory(
        train_dir, 
        subset='training',
        target_size = (image_height, image_width), 
        batch_size = batch_size, 
        class_mode = 'categorical', 
        shuffle = False

    )

    val_data = train_generator.flow_from_directory(
        train_dir,
        subset='validation',
        target_size = (image_height, image_width), 
        batch_size = batch_size, 
        class_mode = 'categorical', 
        shuffle = False
    )

    return train_data, val_data



def build_model(classes_num = 3):

    pretrained_model = tf.keras.applications.VGG16(input_shape = (image_height, image_height, 3), include_top = False, weights = 'imagenet')



    for layer in pretrained_model.layers[:-4]:
        layer.trainable = False


    for layer in pretrained_model.layers:
        print(layer, layer.trainable)

    model = tf.keras.models.Sequential()

    model.add(pretrained_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(classes_num, activation = 'softmax'))

    return model


def compile_model(model, optimzation_algorithm = 'Adam', learning_rate = 0.0001):
    
    if optimzation_algorithm == 'Adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    elif optimzation_algorithm == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',  metrics = ['acc'])

