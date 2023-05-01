import tensorflow as tf
import wandb



class WandbLog:
    """A helper class for logging with wandb"""

    #everything logged during the last step
    log : dict
    def __init__(self) -> None:
        self.log = {}

    @staticmethod
    def wandb_init(project, args, ok_already_initialized = False):
        """Initialize Wandb. Must be called once before any logs are created"""
        if wandb.run is not None:
            if ok_already_initialized:
                return
            raise RuntimeError("Wandb is already initialized")
        wandb.init(project=project, config=args)
        wandb.run.log_code(".")

    def log_image(self, name, image):
        """Log one image to wandb"""
        self.log[name] = wandb.Image(image)
        return self

    def log_segmentation(self, name, image, segmentation, class_labels=None):
        """Log a segmented image to wandb"""
        self.log[name] = wandb.Image(image, masks={
            "predictions":{
                "mask_data":segmentation} | ({} if class_labels is None else dict(enumerate(class_labels)))
        })
        return self

    def log_images(self, name, images):
        """Log a batch of images to wanbd"""
        self.log[name] = [wandb.Image(i) for i in images]
        return self

    def log_value(self, name, value):
        """Log a single value to wandb"""
        self.log[name] = value
        return self

    def log_dict(self, dictionary):
        """Log a dictionary of values / wandb objects"""
        self.log |= dictionary
        return self

    def commit(self):
        """Log everything & upload it to server, where it can be viewed"""
        wandb.log(self.log)
        self.log = {}


class TrainingLog(tf.keras.callbacks.Callback):
    """For logging model training progress, running validation after a given number of batches, and custom model testing"""
    def __init__(self, dev_dataset, test_dataset, log_frequency, test_frequency, validation_frequency=None, train_batch_size = None, learning_rate_decay = None):
        super().__init__()
        self.dev_dataset = dev_dataset
        self.test_dataset = iter(test_dataset.repeat(None))

        self.log_frequency = log_frequency
        self.test_frequency = test_frequency
        self.val_frequency = validation_frequency if validation_frequency is not None else test_frequency
        self.batch_size = dev_dataset.element_spec.shape[0] if train_batch_size is None else train_batch_size
        self.batches_processed = 0

        self.log = WandbLog()

        self.learning_rate_decay = learning_rate_decay

    def on_batch_begin(self, batch, logs=None):
        if self.learning_rate_decay is not None:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate_decay(self.images_processed))
        return super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called by tensorflow when batch ends. Check whether something special like validation or testing should happen"""
        #if we should run the test, get the next piece of data to run it on, and run it
        if self.batches_processed % self.test_frequency == 0:
            data = next(self.test_dataset)
            self.run_test(data)

        #if we should perform validation, do it
        if self.batches_processed % self.val_frequency == 0:
            self.log.log_dict({("val_"+name):val for name, val in self.run_validation().items()})

        #if metrics should be logged
        if self.batches_processed % self.log_frequency == 0:
            #log how many examples were processed until now - this is useful when comparing multiple runs with different batch sizes
            self.log.log_value("images_processed", self.images_processed)
            #add all metrics to the log
            if logs is not None:
                self.log.log_dict(logs)
            #log everything
            self.log.commit()
        self.batches_processed += 1
        
        return super().on_batch_end(batch, logs)

    @property
    def images_processed(self):
        return self.batches_processed*self.batch_size

    def run_test(self, data):
        """Run a test - this can be overriden by child classes to perform testing for different models"""

    def run_validation(self):
        """Evaluate the model using the `self.dev_dataset` dataset, return the metrics"""
        return self.model.evaluate(self.dev_dataset, return_dict=True, verbose=0)
