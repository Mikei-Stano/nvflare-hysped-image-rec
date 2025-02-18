import tensorflow as tf
import pandas as pd
from tf_net import TFNet
import nvflare.client as flare

WEIGHTS_PATH = "./tf_model.weights.h5"
DATASET_PATH = "/workspace/nvflare-runs/simulator-run-convd/smaller_dataset.csv"  # Update this path as needed

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    # Drop the first column and any unwanted columns
    df_encoded = df.drop(columns=[df.columns[0]])
    X = df_encoded.drop(columns=["DRUH_POVR", "NAZ_LOKALI"])
    y = df_encoded["DRUH_POVR"]

    # Convert features to tensor and add a channel dimension for Conv1D
    X_tensor = tf.convert_to_tensor(X.to_numpy(), dtype=tf.float32)
    X_tensor = tf.expand_dims(X_tensor, axis=-1)  # New shape: (samples, num_features, 1)

    # Label encoding using StringLookup
    label_lookup = tf.keras.layers.StringLookup(output_mode='int', vocabulary=tf.constant(y.unique()))
    y_tensor = label_lookup(y) - 1  # Adjust labels to start at 0

    # Determine number of classes
    num_classes = len(label_lookup.get_vocabulary()) - 1

    # One-hot encode labels
    y_one_hot = tf.one_hot(y_tensor, depth=num_classes)

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_one_hot))
    dataset_size = len(y)  # Number of samples
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)

    # Split dataset: 70% train, 15% validation, 15% test
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)

    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)

    # Batch datasets
    batch_size = 32
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset, num_classes

def main():
    flare.init()

    sys_info = flare.system_info()
    print(f"System info: {sys_info}", flush=True)

    # Load data and determine input/output shapes
    train_dataset, val_dataset, test_dataset, num_classes = load_and_preprocess_data()
    # Determine input shape from the dataset (excluding the batch dimension)
    input_shape = train_dataset.element_spec[0].shape[1:]
    
    # Initialize and compile the model with the new architecture
    model = TFNet(input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    model.summary()

    while flare.is_running():
        input_model = flare.receive()
        print(f"Current round: {input_model.current_round}")

        # Update model weights from the global model parameters.
        # We match layers by the names assigned in tf_net.py.
        for k, v in input_model.params.items():
            try:
                layer = model.get_layer(name=k)
                layer.set_weights(v)
            except ValueError:
                print(f"Layer {k} not found in model. Skipping update for this layer.")

        # Evaluate the global model
        loss, test_global_acc = model.evaluate(test_dataset, verbose=2)
        print(f"Global model accuracy on round {input_model.current_round}: {test_global_acc * 100:.2f}%")

        # Train locally for one epoch
        model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=1)
        print("Finished local training.")

        # Save weights for debugging or inspection
        model.save_weights(WEIGHTS_PATH)

        # Prepare and send updated model parameters (only sending layers with weights)
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers if layer.weights},
            params_type="FULL",
            metrics={"categorical_accuracy": test_global_acc},
            current_round=input_model.current_round,
        )
        flare.send(output_model)

if __name__ == "__main__":
    main()
