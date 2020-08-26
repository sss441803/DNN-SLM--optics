import os
from library import use_gpu, model_dir, input_x, input_y, train_total, validation_total, num_of_terms
# This allows libaries to load without using GPU if desired
if use_gpu == False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import Tensor, keras
from tensorflow.keras.layers import Input, Conv2D, ReLU, Dropout, BatchNormalization, \
    Add, AveragePooling2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from generator import training_generator, validation_generator

# This callback function is used to display loss changes during training
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.clf()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show(block=False)
        plt.pause(0.001);

# Loss plotting callback
plot_losses = PlotLosses()

# Defines a batch normalization layer with relu activation
def relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

# Defines a residual block
def residual_block(x: Tensor, filters: int, kernel_size: int = 3) -> Tensor:
    y = relu_bn(x)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(x)
    y = Add()([x, y])
    y = MaxPooling2D(2)(y)
    x = y
    y = relu_bn(x)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(x)
    y = Add()([x, y])
    y = MaxPooling2D(2)(y)
    x = y
    y = relu_bn(x)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(x)
    y = Add()([x, y])
    out = AveragePooling2D(2)(y)
    return out

# Defines model creation
def create_model():
    picture_dimensions = (input_x, input_y, 2)
    inputs = Input(shape=picture_dimensions)
    t = Conv2D(kernel_size=3, strides=1, filters=64, padding="same")(inputs)
    t = residual_block(t, filters=64)
    t = Flatten()(t)
    t = Dense(512, activation= 'relu')(t)
    t = ReLU()(t)
    t = Dropout(0.1)(t)
    t = Dense(512, activation='relu')(t)
    t = ReLU()(t)
    t = Dropout(0.1)(t)
    outputs = Dense(num_of_terms - 2, activation= 'linear')(t)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model

# If the model was already created before, load the model
if os.path.exists(model_dir):
    model = keras.models.load_model(model_dir)
else:
    model = create_model()

# Collect all the callbacks to use during training
my_callbacks = [
    # Stops trainig if no improvement after patience epochs
    keras.callbacks.EarlyStopping(patience=7),
    # Defines how and when to save model
    keras.callbacks.ModelCheckpoint(filepath=model_dir, monitor = 'val_loss', save_best_only = True, verbose=1, mode= 'min', save_weights_only = False),
    # Loss plotting
    plot_losses,
]

# Function to train the model
def fit():
    res = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        callbacks=my_callbacks,
                        verbose = 1,
                        epochs = 200
                        )
    return res

# Fit the model
history = fit()