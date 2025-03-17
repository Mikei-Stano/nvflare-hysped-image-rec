import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class TFNet(models.Sequential):
    def __init__(self, input_shape=(474, 1), num_classes=18):
        super().__init__()
        # Input layer with explicit name
        self.add(layers.Input(shape=input_shape, name="input_layer"))
        # Batch Normalization layer
        self.add(layers.BatchNormalization(name="bn_input"))
        
        # Smaller architecture: fewer layers, smaller filters, and kernel sizes
        filters = [64, 32, 16]
        kernel_sizes = [3, 3, 3]
        padding = 'same'  # Use "same" padding to keep the output length unchanged

        # TODO: Bigger Convolutional blocks configuration
        # filters = [1024, 512, 256, 128]
        # kernel_sizes = [9, 9, 9, 9]
        # padding = 'causal'
        
        # Add convolutional blocks with explicit names for weight mapping
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            self.add(layers.Conv1D(
                filters=f,
                kernel_size=k,
                activation='relu',
                padding=padding,
                kernel_regularizer=regularizers.l2(),
                name=f"conv_{i+1}"
            ))
            self.add(layers.BatchNormalization(name=f"bn_{i+1}"))
            self.add(layers.ReLU(name=f"relu_{i+1}"))
        
        # Global Average Pooling and output Dense layer
        self.add(layers.GlobalAveragePooling1D(name="global_avg_pool"))
        self.add(layers.Dense(num_classes, activation='softmax', name="dense_out"))