from keras.callbacks import Callback, EarlyStopping
import timeit


class TrainLogger(Callback):
    """
    Provides methods to write current loss and validation results to log file during training.
    """

    def __init__(self, log_file):
        super(TrainLogger, self).__init__()
        self.log_file = log_file
        # overwrite old file
        with open(log_file, 'w') as f:
            pass

    def on_epoch_end(self, epoch, logs=None):
        self.write_loss(epoch, float(logs['loss']))

    def write_loss(self, epoch, loss):
        s = 'epoch {} - loss {}'.format(epoch, float(loss))
        self.write(s)

    def write_validation(self, epoch, value):
        s = 'epoch {} - valid {}'.format(epoch, float(value))
        self.write(s)

    def write(self, string):
        with open(self.log_file, 'a') as f:
            f.write(string + '\n')


class BatchStatusCallback(Callback):
    """
    Show loss and time left during epoch for given amount of batches.
    """

    def __init__(self, sample_per_epoch, batch_size, updates=10):
        super(BatchStatusCallback, self).__init__()
        self.batch_size = batch_size
        if updates >= 1:
            self.update_per_num_batches = int((sample_per_epoch / self.batch_size / updates) + 1)
        else:
            self.update_per_num_batches = sample_per_epoch + 1
        self.batches = int(sample_per_epoch / self.batch_size)
        self.epoch_start_time = 0

    def on_batch_end(self, batch, logs={}):
        batch += 1
        time_passed = timeit.default_timer() - self.epoch_start_time
        time_per_batch = time_passed / batch
        time_left = int(time_per_batch * (self.batches - batch))
        # condition can't be True if {updates} < 1; See constructor
        if batch % self.update_per_num_batches == 0:
            self.print_status(batch, time_left, float(logs['loss']))

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = timeit.default_timer()

    def print_status(self, batch, time_left, loss):
        print('    Batch {0}/{1} - loss: {2:.5f} - time_left: {3}s'.format(batch, self.batches, loss, time_left))


class EarlyStoppingAndSaving(EarlyStopping):

    def __init__(self, output_file, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStoppingAndSaving, self).__init__(monitor, min_delta, patience, verbose, mode)
        self.output_file = output_file

    def on_epoch_end(self, epoch, logs=None):
        super(EarlyStoppingAndSaving, self).on_epoch_end(epoch, logs)
        # if the current loss is also the best, then save weights
        if self.best == logs.get(self.monitor):
            self.model.save_weights(self.output_file, overwrite=True)
            print('wrote model weights (with best {}) to {}'.format(self.monitor, self.output_file))



