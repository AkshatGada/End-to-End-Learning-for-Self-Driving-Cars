import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=False):
   
    shortcut = x
    if use_conv_shortcut:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet(input_shape=(66, 200, 3), num_outputs=1):
    """
    Builds a simple ResNet-like model using residual blocks.
    
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_outputs (int): Number of outputs. For regression (steering angle prediction), this is 1.
    
    Returns:
        Model: A Keras Model instance.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution and max pooling
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual Block Group 1
    x = residual_block(x, filters=64, kernel_size=3, strides=1, use_conv_shortcut=True)
    x = residual_block(x, filters=64, kernel_size=3, strides=1)
    
    # Residual Block Group 2
    x = residual_block(x, filters=128, kernel_size=3, strides=2, use_conv_shortcut=True)
    x = residual_block(x, filters=128, kernel_size=3, strides=1)
    
    # Residual Block Group 3
    x = residual_block(x, filters=256, kernel_size=3, strides=2, use_conv_shortcut=True)
    x = residual_block(x, filters=256, kernel_size=3, strides=1)
    
    # (Optional) Residual Block Group 4 for deeper models
    x = residual_block(x, filters=512, kernel_size=3, strides=2, use_conv_shortcut=True)
    x = residual_block(x, filters=512, kernel_size=3, strides=1)
    
    # Global average pooling to reduce spatial dimensions
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layers for regression
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(num_outputs)(x)  # Linear activation is used for regression by default
    
    model = Model(inputs, outputs)
    return model

# Build and compile the ResNet model
model = build_resnet()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

# Display the model architecture
model.summary()
