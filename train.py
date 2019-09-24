from model import *
from callbacks import *


training_dir = 'imagecl/train'
epochs_num = 90


training_data, validation_data = import_data(train_dir = training_dir)

model = build_model()

compile_model(model, optimzation_algorithm = 'Adam', learning_rate = 0.0001)


history = model.fit_generator(training_data,
            validation_data = validation_data,
            steps_per_epoch = 25,
            epochs = epochs_num,
            validation_steps = 6, 
            shuffle = True, 
            verbose = 2, 
            callbacks=callbacks)

