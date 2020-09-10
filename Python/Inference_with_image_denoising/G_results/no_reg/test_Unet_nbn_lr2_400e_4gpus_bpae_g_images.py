import numpy as np
import os
import sys
import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential

import random
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow.keras as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D, LeakyReLU
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#devices list
from tensorflow.python.client import device_lib

#import pickle
import pickle
MODEL_SAVE_NAME = 0
MODEL_SAVE_WEIGHTS = 1
#import warnings
from tensorflow.keras.callbacks import Callback
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"#",1,2,3"  # specify which GPU(s) to be used
import json
from pprint import pprint
import scipy.io as io
from PIL import Image
import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize

test_set = 8 #G_images

CSV_path = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Spring20/Jan20/1101/Keras_inference_bpae_cells_G_images/'
class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, num_exps=100, batch_size=4, dim=(512,512), n_channels=1, shuffle=True, train = False, validation = False, test = False, test_set =1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.num_exps = num_exps
        self.train = train
        self.validation = validation
        self.test = test
        self.test_set = test_set
        self.captures = 50
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_exps / self.batch_size))

    def __getitem__(self, index): #index = batch num
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = indexes#[self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_exps)
        if self.train == True:
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(((self.batch_size), *self.dim, self.n_channels)) #total  batch size here includes slices
        Y = np.empty(((self.batch_size), *self.dim, self.n_channels))
        if self.train == True:
            df = pandas.read_csv(CSV_path + 'samples_train.csv')
        if self.validation == True:
            df = pandas.read_csv(CSV_path + 'samples_test.csv')
        if self.test == True:
            if self.test_set==1:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_raw1.csv')
            elif self.test_set==2:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg2.csv')
            elif self.test_set==3:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg4.csv')
            elif self.test_set==4:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg8.csv')
            elif self.test_set==5:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg16.csv')
            elif self.test_set==6:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_all_grouped_repeat.csv')
            elif self.test_set==7:
                df2 = pandas.read_csv(CSV_path + 'test_lifetime_samples_raw_bpae.csv') #lifetime images mouse_kidney
            else:
                df2 = pandas.read_csv(CSV_path + 'test_samples_raw_bpae_reg_g_images.csv') #G_images reg
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.train == True or self.validation == True:
                files1 = df['Images'][ID*self.captures:(ID+1)*self.captures]
                indx1 = range(ID*self.captures,(ID+1)*self.captures)
                indx1 = np.array(indx1) #to array
                #take input from id*50 to (id+1)*50, shuffle the data and take input0 and input1 as input and target
                if self.shuffle == True:
                    np.random.shuffle(indx1) #shuffle array
                fx1 = indx1[0] #input 
                fx2 = indx1[1] #target
                
                img = load_img(df['Images'][fx1], color_mode = "grayscale")
                label = load_img(df['Images'][fx2] , color_mode = "grayscale")

            if self.test == True: #test no shuffle
            #from skimage import io
                from skimage import io
                #img = load_img(df2['Images'][ID], color_mode = "grayscale")
                #label = load_img(df2['Labels'][ID] , color_mode = "grayscale")
                #img1 = img_to_array(img)
                #label1 = img_to_array(label)
                img = io.imread(df2['Images'][ID]) #reading from a tif file
                label = io.imread(df2['Labels'][ID]) #reading from a tif file
                min_G = -0.1
                max_G = 1.1
                img = np.clip(img, min_G, max_G)
                label = np.clip(label, min_G, max_G)
                img = (img+0.1)/1.2 #g range is -0.1 to 1.1 (so add 0.1 range -> 0 to 1.2 and divide by 1.2 to bring 0 to 1 range)
                label = (label+0.1)/1.2 #g range is -0.1 to 1.1 (so add 0.1 range -> 0 to 1.2 and divide by 1.2 to bring 0 to 1 range)
            
            x_img = img_to_array(img)
            y_label = img_to_array(label)
            #print('max of input',x_img[0:5][0:5])
            #x_img = resize(x_img, (512, 512, 1), mode='constant', preserve_range=True) #actual  image and label
            #y_label = resize(y_label, (512, 512, 1), mode='constant', preserve_range=True)#actual  image and label
            
            #imx1 = np.array([x_img[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,x_img.shape[0],self.dim[0]) for y in range(0,x_img.shape[1],self.dim[1])])
            #lbx1 = np.array([y_label[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,y_label.shape[0],self.dim[0]) for y in range(0,y_label.shape[1],self.dim[1])])
            
            #X[i:(i+1), ..., 0] = x_img.squeeze() / 255 - 0.5
            X[i:(i+1), ..., 0] = x_img.squeeze() - 0.5
            #Y[i:(i+1), ..., 0] = y_label.squeeze() / 255 - 0.5
            Y[i:(i+1), ..., 0] = y_label.squeeze() - 0.5
            #print('max of input',X[i][0:5][0:5])

        return X, Y
        
def save_weight_light(model, filename):
	layers = model.layers
	pickle_list = []
	for layerId in range(len(layers)):
		weigths = layers[layerId].get_weights()
		pickle_list.append([layers[layerId].name, weigths])

	with open(filename, 'wb') as f:
		pickle.dump(pickle_list, f, -1)

def load_weight_light(model, filename):
	layers = model.layers
	with open(filename, 'rb') as f:
		pickle_list = pickle.load(f)

	for layerId in range(len(layers)):
		assert(layers[layerId].name == pickle_list[layerId][MODEL_SAVE_NAME])
		layers[layerId].set_weights(pickle_list[layerId][MODEL_SAVE_WEIGHTS])


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_gpus =  get_available_gpus()
print('>>>>>>>> Available GPU devices list is here: >>>>>>>>>>',get_gpus)

#multiple GPUS training
#from multi_gpu_utils import multi_gpu_models ##so much dependance -> use latest TF version here
from tensorflow.keras.utils import multi_gpu_model #-> use for next keras release version
# or use this pip install --user git+git://github.com/fchollet/keras.git --upgrade in job file/in terminal that upgrades the keras version
import multiprocessing #for CPU count

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.backend import set_session
import tensorflow.keras.backend as Kb
#Kb._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #either true or per fraction need to be set
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True

t1= time.time()
for d in ['/device:GPU:0']: #, '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
    with tf.device(d):
        batch_size = 1
        im_height = 512 #M
        im_width = 512  #N
        num_exps_test = 50 #test dataset with raw data
        
        #test_set =1 raw data, 2 -> avg2 , 3-> avg4, 4 -> avg8, 5-> avg16, 6-> all together noise levels (1,2,4,8,16)
        params_test = {'dim': (im_height,im_width),'num_exps':num_exps_test,'batch_size':batch_size,'n_channels': 1,'shuffle': False, 'train': False,  'validation': False, 'test': True,'test_set':test_set}
        test_generator = DataGenerator(**params_test)


def get_unet(input_img):
    # contracting path
    #256
    xc0 = Conv2D(filters=48, kernel_size=(3,3), kernel_initializer="he_normal",padding="same",use_bias=False)(input_img)
    #xa0 = Activation("elu")(xc0,0.1)
    xa0 = LeakyReLU(alpha=0.1)(xc0)
    xc1 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal",padding="same",use_bias=False)(xa0)
    #xcbn1 = BatchNormalization()(xc1)
    xa1 = LeakyReLU(alpha=0.1)(xc1)
    #xa1 = Activation("elu")(xcbn1,alpha=0.1)
    xp1 = MaxPooling2D((2, 2)) (xa1)
    #print('DDDDD',xp1.shape)
    
    #128
    xc2 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal",padding="same",use_bias=False)(xp1)
    #xcbn2 = BatchNormalization()(xc2)
    xa2 = LeakyReLU(alpha=0.1)(xc2)
    #xa2 = Activation("elu")(xcbn2,alpha=0.1)
    xp2 = MaxPooling2D((2, 2)) (xa2)
    #print('EEEEEE',xp2.shape)
    #64
    xc3 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal",padding="same",use_bias=False)(xp2)
    #xcbn3 = BatchNormalization()(xc3)
    xa3 = LeakyReLU(alpha=0.1)(xc3)
    #xa3 = Activation("elu")(xcbn3,alpha=0.1)
    xp3 = MaxPooling2D((2, 2)) (xa3)
    #print('NNNNNN',xp3.shape)
    #32
    xc4 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal",padding="same",use_bias=False)(xp3)
    #xcbn4 = BatchNormalization()(xc4)
    xa4 = LeakyReLU(alpha=0.1)(xc4)
    #xa4 = Activation("elu")(xcbn4,alpha=0.1)
    xp4 = MaxPooling2D((2, 2)) (xa4)
    #print('FFFFFF',xp4.shape)
    #16
    xc5 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same",use_bias=False)(xp4)
    #xcbn5 = BatchNormalization()(xc5)
    xa5 = LeakyReLU(alpha=0.1)(xc5)
    #xa5 = Activation("elu")(xcbn5,alpha=0.1)
    xp5 = MaxPooling2D((2, 2)) (xa5)
    #print('GGGGGG',xp5.shape)
    #8
    xc6 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same",use_bias=False)(xp5)
    #xcbn6 = BatchNormalization()(xc6)
    xa6 = LeakyReLU(alpha=0.1)(xc6)
    #xa6 = Activation("elu")(xcbn6,alpha=0.1)
    xup5 = UpSampling2D((2, 2))(xa6)
    #print('HHHHHH',xup5.shape)
    #16

    cat1 = concatenate([xup5, xp4])
    #print('cat1',cat1.shape)
    #xdc5a =  Conv2DTranspose(96, (3, 3), strides=(2, 2), padding='same') (cat1)
    #xdc5a = Conv2DTranspose(96, (3, 3), strides=(2, 2), padding='same') (cat1)
    xdc5a = Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (cat1)
    #print('xdc5a',xdc5a.shape)
    #xdbn5a = BatchNormalization()(xdc5a)
    #print('xdbn5a',xdbn5a.shape)
    xad5a = LeakyReLU(alpha=0.1)(xdc5a)
    #print('xad5a',xad5a.shape)
    #xad5a = Activation("elu")(xdbn5a,alpha=0.1)
    xdc5b =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad5a)
    #print('xdc5b',xdc5b.shape)
    #xdbn5b = BatchNormalization()(xdc5b)
    #print('xdbn5b',xdbn5b.shape)
    xad5b = LeakyReLU(alpha=0.1)(xdc5b)
    #print('xad5b',xad5b.shape)
    #xad5b = Activation("elu")(xdbn5b,alpha=0.1)
    xup4 = UpSampling2D((2, 2))(xad5b)
    #print('xup4',xup4.shape)
    #32
    #print('xup4',xup4.shape)
    cat2 = concatenate([xup4, xp3])
    #print('cat2',cat2.shape)
    xdc4a =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (cat2) #op channels (96) not input channels (144)
    #xdbn4a = BatchNormalization()(xdc4a)
    xad4a = LeakyReLU(alpha=0.1)(xdc4a)
    #xad4a = Activation("elu")(xdbn4a,alpha=0.1)
    xdc4b =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad4a)
    #xdbn4b = BatchNormalization()(xdc4b)
    xad4b = LeakyReLU(alpha=0.1)(xdc4b)
    #xad4b = Activation("elu")(xdbn4b,alpha=0.1)
    xup3 = UpSampling2D((2, 2))(xad4b)
    #print('KKKKKK',xup3.shape)
    #64

    cat3 = concatenate([xup3, xp2])
    xdc3a =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (cat3) #op channels (96) not input channels (144)
    #xdbn3a = BatchNormalization()(xdc3a)
    xad3a = LeakyReLU(alpha=0.1)(xdc3a)
    #xad3a = Activation("elu")(xdbn3a,alpha=0.1)
    xdc3b =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad3a)
    #xdbn3b = BatchNormalization()(xdc3b)
    xad3b = LeakyReLU(alpha=0.1)(xdc3b)
    #xad3b = Activation("elu")(xdbn3b,alpha=0.1)
    xup2 = UpSampling2D((2, 2))(xad3b)
    #print('UUUUUU',xup2.shape)
    #128

    cat4 = concatenate([xup2, xp1])
    xdc2a =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (cat4) #op channels (96) not input channels (144)
    #xdbn2a = BatchNormalization()(xdc2a)
    xad2a = LeakyReLU(alpha=0.1)(xdc2a)
    #xad2a = Activation("elu")(xdbn4a,alpha=0.1)
    xdc2b =  Conv2D(96, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad2a)
    #xdbn2b = BatchNormalization()(xdc2b)
    xad2b = LeakyReLU(alpha=0.1)(xdc2b)
    #xad2b = Activation("elu")(xdbn4b,alpha=0.1)
    xup1 = UpSampling2D((2, 2))(xad2b)
    #print('YYYYYY',xup1.shape)
    #256

    cat5 = concatenate([xup1, input_img])
    xdc1a =  Conv2D(64, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (cat5) #op channels (64) not input channels (96+1: 1 for input_channels)
    #xdbn1a = BatchNormalization()(xdc1a)
    xad1a = LeakyReLU(alpha=0.1)(xdc1a)
    #xad1a = Activation("elu")(xdbn1a,alpha=0.1)
    xdc1b =  Conv2D(32, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad1a) #input 64, output=32
    #xdbn1b = BatchNormalization()(xdc1b)
    xad1b = LeakyReLU(alpha=0.1)(xdc1b)
    #xad1b = Activation("elu")(xdbn1b,alpha=0.1)
    xdc1c =  Conv2D(1, (3, 3), kernel_initializer="he_normal", padding='same',use_bias=False) (xad1b) #input 32, output=1
    outputs = Activation("tanh")(xdc1c)

    #model = Model(inputs=[input_img], outputs=[outputs])
    #return model
    return outputs


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count = shapes_mem_count+single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def test():
    
    input_img = Input((512, 512, 1), name='img')
    output_img = Input((512, 512, 1), name='label')
    outputs = get_unet(input_img)
    model = Model(inputs=[input_img],outputs=[outputs])
    
    import pickle
    load_weight_light(model,'model-unet_multiple_gpus.h5')
    print('>>>>>>>  Model summary is here >>>>>>',model.summary())
    
    #gbytes_model = get_model_memory_usage (batch_size=batch_size, model=model)
    #print('>>>>>>> Memory used in the Model: (GBytes) >>>>>>>>>>',gbytes_model)
        
    if len(get_gpus) <=1:
        print(" >>>>>> training with 1 GPU >>>>>> ")
        model.build(input_img)
    else:
        print(" >>>>>>>> training with {} GPUs >>>>>>".format(len(get_gpus)))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/device:CPU:0"):
            # initialize the model
            model.build(input_img)
            #print('<<<< Model is here <<<<<',model)
            # make the model parallel
        model = multi_gpu_model(model, gpus=len(get_gpus))
        
    #model compile
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=["mse"])
    #print('<<<< Model summary is here <<<<<',model.summary())
    tx12 = time.time()
    test_results = model.predict_generator(generator=test_generator, steps=None, use_multiprocessing=True, workers=4, max_queue_size=10)
    tx13 = time.time()
    print('inference Execution time is: \n',tx13-tx12)
    import scipy.io as io
    print('result type ', test_results.shape)
    io.savemat('test_nbn_Estimated_result_G_not_registered_bpae_512.mat', dict([('test_nbn_Estimated_result_G_not_registered_bpae_512',test_results)]))
    test_results_final_epoch = model.evaluate_generator(generator=test_generator, steps=None, max_queue_size=10, workers=4, use_multiprocessing=False)
    
    return test_results_final_epoch
    
    
if __name__ == "__main__":
    Epochs =  400 #used epochs in train function
    test_results_final_epoch = test()
    print('>>>> test_results (loss and MSE on final epoch)are here >>>> ', test_results_final_epoch)
    t2 = time.time()
    print('Execution time is: \n',t2-t1)