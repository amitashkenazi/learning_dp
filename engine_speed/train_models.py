import json
import glob
from service_functions import prepare_dataset, batch_generator
import pickle
import os.path
import time
from keras.models import Sequential, load_model
from keras.layers import InputLayer, MaxPooling1D, Conv1D, Dense, Flatten #, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model
from keras.optimizers import Adam

import os

dumpfile_datasets = "datasets.pickle"
all_files = glob.glob("soundfiles/*.wav")

sound_interval_ms=300.
sound_stride_ms=150.
time_per_file = 5000.
batch_size = 128
number_of_samples = int(time_per_file/sound_stride_ms) * len(all_files)
steps_per_epoch = int(number_of_samples/batch_size)

if os.path.isfile(dumpfile_datasets) :
    with open(dumpfile_datasets, 'rb') as handle:
        input_shape, train_files, x_validation, y_validation, x_test, y_test = pickle.load(handle)    
else:
    input_shape, train_files, x_validation, y_validation, x_test, y_test = prepare_dataset(all_files, sound_interval_ms=sound_interval_ms, sound_stride_ms=sound_stride_ms)
    all_datasets = (input_shape, train_files, x_validation, y_validation, x_test, y_test)
    with open(dumpfile_datasets, 'wa') as handle:
        pickle.dump(all_datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


generator = batch_generator(batch_size=batch_size, train_files=train_files, sound_interval_ms=sound_interval_ms, sound_stride_ms=sound_stride_ms)

filter1_sizes = [[20,50], [50,100]]
filter1_strides = [1,5]
filter2_sizes = [[20,100], [50,50]]
filter2_strides = [1,5]
dense_sizes = [64,128,256]
pool1_sizes_strides = [[15, 15], [25,25]]
pool2_sizes_strides = [[15, 15], [25,25]]
lrs = [1e-3, 1e-4]
idx = 0
start_test_time = int(time.time())
test_dir = "test_{}".format(start_test_time)
os.makedirs(test_dir)
results = {}
for filter1_size in filter1_sizes:
    for filter1_stride in filter1_strides:
        for filter2_size in filter2_sizes:
            for filter2_stride in filter2_strides:
                for dense_size in dense_sizes:
                    for pool1_sizes_stride in pool1_sizes_strides:
                        for pool2_sizes_stride in pool2_sizes_strides:
                            for lr in lrs:
                                idx += 1
                                current_test_dir = os.path.join(test_dir, str(idx))
                                os.makedirs(current_test_dir)
                                
                                try:
                                    meta_dict = {"filter1_size":filter1_size, "filter1_stride":filter1_stride, "filter2_size":filter2_size, "filter2_stride":filter2_stride, "dense_size":dense_size, "pool1_sizes_stride":pool1_sizes_stride, "pool1_sizes_stride":pool1_sizes_stride, "lr":lr}
                                    with open(os.path.join(current_test_dir, 'metadata.json'), 'w') as outfile:
                                        print json.dumps(meta_dict)
                                        json.dump(meta_dict, outfile)
                                    model = Sequential()
                                    model.add(InputLayer(batch_input_shape=input_shape))
                                    model.add(Conv1D(filters=filter1_size[0], kernel_size=filter1_size[1] ,strides=filter1_stride, activation= 'relu', name="conv1"))
                                    model.add(MaxPooling1D(pool_size=pool1_sizes_stride[0], strides=pool1_sizes_stride[1]))
                                    model.add(Conv1D(filters=filter2_size[0], kernel_size=filter2_size[1] ,strides=filter2_stride, activation= 'relu', name="conv3")) 
                                    model.add(MaxPooling1D(pool_size=pool2_sizes_stride[0], strides=pool2_sizes_stride[1]))
                                    model.add(Flatten())
                                    model.add(Dense(dense_size, activation='relu'))
                                    # model.add(Dropout(0.5))
                                    model.add(Dense(1, activation='relu'))

                                    print(model.summary())
                                    plot_model(model, to_file=os.path.join(current_test_dir, 'model_plot.png'), show_shapes=True, show_layer_names=True)

                                    optimizer = Adam(lr=lr)

                                    model.compile(optimizer=optimizer,
                                                  loss='mean_squared_error',
                                                  metrics=['accuracy'])

                                    if os.path.isfile("test_1527018883/{}/keras_1527018884".format(idx)):
                                        print "------------loading weights---------------"
                                        model.load_weights("test_1527018883/{}/keras_1527018884".format(idx))
                                        print "Done"
                                    else:
                                        print "----------not loading weights-------------"

                                    callback_reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)
                                    callback_tensorboard = TensorBoard(log_dir=os.path.join(current_test_dir, 'engine_raw_singal_logs/'), histogram_freq=0, write_graph=False)
                                    callback_early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
                                    path_checkpoint = os.path.join(current_test_dir, 'keras_{}'.format(int(time.time())))
                                    checkpointer = ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_best_only=True)

                                    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                                                          monitor='loss',
                                                                          verbose=1,
                                                                          save_weights_only=True,
                                                                          save_best_only=True)
                                    callbacks = [ callback_early_stopping, callback_checkpoint, callback_reduce_lr ]

                                    model_json = model.to_json()
                                    with open(os.path.join(current_test_dir, "model_raw.json"), "w") as json_file:
                                        json_file.write(model_json)
                                    
                                    history = model.fit_generator(generator=generator,
                                                        validation_data = (x_validation,y_validation),
                                                        epochs=1000,
                                                        steps_per_epoch=steps_per_epoch,
                                                        callbacks=callbacks
                                                       )
                                    result = model.evaluate(x=x_test,
                                                            y=y_test)
                                    results[idx] = result

                                    with open(os.path.join(test_dir, 'results.json'), 'w') as outfile:
                                        json.dump(results, outfile)
                                    
                                    # serialize weights to HDF5
                                    model.save_weights(os.path.join(current_test_dir, "model_raw.h5"))
                                    print("Saved model to disk")
                                except Exception as e:
                                    print e
                                    results[idx] = ["failed"]






