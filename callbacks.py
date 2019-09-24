import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model


accuracy_threshold = 0.99
patience = 50


class callback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('acc') > accuracy_threshold):
            print("Reached %2.2f%% accuracy, Training stopped!!" %(accuracy_threshold*100))
            self.model.stop_training = True
            self.model.save('model-optimal.h5')
        

        
checkpoint = tf.keras.callbacks.ModelCheckpoint('weights{epoch:03d}.h5', save_weights_only=True, period=10)     
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_acc', factor=0.1, patience=10, verbose=1)
callbacks = [callback(), checkpoint]





