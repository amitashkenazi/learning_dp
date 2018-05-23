import soundfile
import numpy as np
import scipy.signal

def read_and_split(filepath):
    samples, fs = soundfile.read(filepath)
    samples = np.array(samples)
    sound_samples = samples[:, 0]
    sound_samples = sound_samples - np.mean(sound_samples)
    tach_samples = samples[:, 4]
    tach_samples = tach_samples - np.mean(tach_samples)
    tach_samples = scipy.signal.medfilt(tach_samples, kernel_size=[5])
    return sound_samples, tach_samples, fs


def calc_speed(tach_samples, fs, number_elements, samples_for_claculation=3000):
    positive = np.where(tach_samples > 0)[0]
    positive_roll = np.roll(positive, 1)
    sub = positive - positive_roll
    sub[0] = 1
    cross_positive = positive[np.where(sub!=1)[0]]
    negative = np.where(tach_samples < 0)[0]
    negative_roll = np.roll(negative, 1)
    sub = negative - negative_roll
    sub[0] = 1
    cross_negative = negative[np.where(sub!=1)[0]]
    cross_vector = np.zeros(len(tach_samples))
    cross_vector[cross_positive] = 1
    cross_vector[cross_negative] = 1
    neg_pos_concat = np.concatenate((np.diff(cross_positive), np.diff(cross_negative)))
    speed = 1 / ((np.mean(neg_pos_concat)/ fs) * number_elements)
    return speed, tach_samples, cross_vector


def split_datasets(file_list, train=0.6, validation=0.2, test=.2):
	assert train+validation+test == 1
	train_files_idx_start = 0
	train_files_idx_end = int(train * len(file_list))
	validation_files_idx_start = train_files_idx_end
	validation_files_idx_end = validation_files_idx_start + int(validation * len(file_list))
	test_files_idx_start = validation_files_idx_end
	test_files_idx_end = test_files_idx_start + int(test * len(file_list)) 
	train_files = file_list[train_files_idx_start:train_files_idx_end]
	validation_files = file_list[validation_files_idx_start:validation_files_idx_end]
	test_files = file_list[test_files_idx_start:test_files_idx_end]

	return train_files, validation_files, test_files



def prepare_dataset(file_list, sound_interval_ms, sound_stride_ms, batch_size=128, expected_fs=48000):

	sound_interval_samples = int(expected_fs * sound_interval_ms/1000)
	input_shape = (None, sound_interval_samples, 1)
	train_files, validation_files, test_files = split_datasets(file_list, train=.6, validation=.2, test=.2)

	x_validation = []
	y_validation = []
	x_test = []
	y_test = []
	# load test and validation
	print "processing validation files"
	for f in validation_files:
	    print ".",
	    sound_samples, tach_samples, fs = read_and_split(f)
	    assert expected_fs == fs
	    sound_interval_samples = int(fs*sound_interval_ms/1000)
	    sound_stride_samples = int(fs*sound_stride_ms/1000)
	    num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
	    for n in range(int(num_of_strides)):
	        interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
	        x_validation.append(interval_samples)
	        y_validation.append(speed)
	print ""
	print "processing test files"
	for f in test_files:
	    print ".",
	    sound_samples, tach_samples, fs = read_and_split(f)
	    assert expected_fs == fs
	    sound_interval_samples = int(fs*sound_interval_ms/1000)
	    sound_stride_samples = int(fs*sound_stride_ms/1000)
	    num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
	    for n in range(int(num_of_strides)):
	        interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
	        x_test.append(interval_samples)
	        y_test.append(speed)
	print ""
	x_validation = np.array(x_validation)
	y_validation = np.array(y_validation)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	x_validation = np.expand_dims(x_validation, axis=2)
	x_test = np.expand_dims(x_test, axis=2)
	print x_validation.shape, y_validation.shape
	print x_test.shape, y_test.shape
	print "Done"
	return input_shape, train_files, x_validation, y_validation, x_test, y_test

def batch_generator(batch_size, train_files, sound_interval_ms, sound_stride_ms, expected_fs=48000):
    """
    Generator function for creating random batches of training-data.
    """
    x_batch = []
    y_batch = []
    while True:
        for f in train_files:
            sound_samples, tach_samples, fs = read_and_split(f)
            assert expected_fs == fs
            sound_interval_samples = int(fs*sound_interval_ms/1000)
            sound_stride_samples = int(fs*sound_stride_ms/1000)
            num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
            for n in range(int(num_of_strides)):
                interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
                interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
                speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
                x_batch.append(interval_samples)
                y_batch.append(speed)
                if len(x_batch) >= batch_size:
                    x_batch = np.expand_dims(x_batch, axis=2)
                    yield (np.array(x_batch), np.array(y_batch))
                    x_batch = []
                    y_batch = []


def prepare_dataset_time_sequence(file_list, sound_interval_ms, sound_stride_ms, batch_size=128, expected_fs=48000, num_sequencial_samples=None):

	sound_interval_samples = int(expected_fs * sound_interval_ms/1000)
	input_shape = (None, sound_interval_samples, 1)
	train_files, validation_files, test_files = split_datasets(file_list, train=.6, validation=.2, test=.2)

	x_validation = []
	y_validation = []
	x_test = []
	y_test = []
	# load test and validation
	print "processing validation files"
	intervals_samples = []
	for f in validation_files:
	    print ".",
	    sound_samples, tach_samples, fs = read_and_split(f)
	    assert expected_fs == fs
	    sound_interval_samples = int(fs*sound_interval_ms/1000)
	    sound_stride_samples = int(fs*sound_stride_ms/1000)
	    num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
	    for n in range(int(num_of_strides)):
	        interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        if num_sequencial_samples is not None:
	        	intervals_samples.append(interval_samples)
	        	if len(intervals_samples) < num_sequencial_samples:
	        		continue
	        	else:
	        		x_validation_sample = intervals_samples
	        		intervals_samples.pop(0)
        	else:
        		x_validation_sample = interval_samples

	        interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
	        x_validation.append(x_validation_sample)
	        y_validation.append(speed)
	print ""
	print "processing test files"
	intervals_samples = []
	for f in test_files:
	    print ".",
	    sound_samples, tach_samples, fs = read_and_split(f)
	    assert expected_fs == fs
	    sound_interval_samples = int(fs*sound_interval_ms/1000)
	    sound_stride_samples = int(fs*sound_stride_ms/1000)
	    num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
	    for n in range(int(num_of_strides)):
	        interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        if num_sequencial_samples is not None:
	        	intervals_samples.append(interval_samples)
	        	if len(intervals_samples) < num_sequencial_samples:
	        		continue
	        	else:
	        		x_test_sample = intervals_samples
	        		intervals_samples.pop(0)
	        else:
        		x_test_sample = interval_samples

	        interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
	        speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
	        x_test.append(x_test_sample)
	        y_test.append(speed)
	print ""
	x_validation = np.array(x_validation)
	y_validation = np.array(y_validation)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	

	if num_sequencial_samples is not None:
		x_validation = np.expand_dims(x_validation, axis=2)
		x_test = np.expand_dims(x_test, axis=2)
	else:
		x_validation = np.expand_dims(x_validation, axis=3)
		x_test = np.expand_dims(x_test, axis=3)


	print x_validation.shape, y_validation.shape
	print x_test.shape, y_test.shape
	print "Done"
	return input_shape, train_files, x_validation, y_validation, x_test, y_test

def batch_generator_time_sequence(batch_size):
    """
    Generator function for creating random batches of training-data.
    """
    x_batch = []
    y_batch = []
    x_intervals = []
    while True:
        for f in train_files:
            sound_samples, tach_samples, fs = read_and_split(f)
            assert expected_fs == fs
            sound_interval_samples = int(fs*sound_interval_ms/1000)
            sound_stride_samples = int(fs*sound_stride_ms/1000)
            num_of_strides = (len(sound_samples)-sound_interval_samples)/sound_stride_samples
            for n in range(int(num_of_strides)):
                interval_samples = sound_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
                interval_tach = tach_samples[n*sound_stride_samples:n*sound_stride_samples+sound_interval_samples]
                speed, tach_samples_partial,cross_vector = calc_speed(interval_tach, fs, number_elements=20)
                x_intervals.append(interval_samples)
                if len(x_intervals) < 11:
                    continue
                else:
                    x_batch.append(x_intervals)
                    x_intervals.pop(0)
                    y_batch.append(speed)
                    if len(x_batch) >= batch_size:
                        x_batch = np.expand_dims(x_batch, axis=3)
                        yield (np.array(x_batch), np.array(y_batch))
                        x_batch = []
                        y_batch = []