from tensorflow.keras import layers, models

class TFNet(models.Sequential):
    def __init__(self, input_shape=(474,), num_classes=18):
        super().__init__()
        self.add(layers.InputLayer(input_shape=input_shape))  # Input layer
        self.add(layers.Dense(64, activation="relu"))  # First Dense layer
        self.add(layers.Dense(num_classes, activation="softmax"))  # Output layer for classification
