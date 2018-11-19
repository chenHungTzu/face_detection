import traning_module_CNN
import traning_file_structure
import parameter as e
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,     
                                   zoom_range = 0.2,      
                                   horizontal_flip = True 
                                  )
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(
    e.train_path_prefix + '/', target_size = (64, 64),
    batch_size = e.batch_size,
    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
    e.test_path_prefix + '/', target_size = (64, 64),
    batch_size = e.batch_size, 
    class_mode = 'categorical')


history = traning_module_CNN.classifier.fit_generator(training_set,
                         nb_epoch=e.nb_epoch,
                         nb_val_samples=e.nb_val_samples,
                         steps_per_epoch = e.steps_per_epoch,
                         verbose = e.verbose,
                         validation_data = test_set)

