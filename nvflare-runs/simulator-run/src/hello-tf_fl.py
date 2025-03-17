import tensorflow as tf
import pandas as pd
from tf_net import TFNet

import nvflare.client as flare

WEIGHTS_PATH = "./tf_model.weights.h5"
DATASET_PATH = "/home/pocik/Documents/tutorials/nvflare-hysped-image-rec/first-experiment/smaller_dataset.csv" # Change this to match your location

# Custom preprocessing logic
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Preprocess the dataset
    df_encoded = df.drop(columns=[df.columns[0]])  # Drop the first column
    X = df_encoded.drop(columns=["DRUH_POVR", "NAZ_LOKALI"])  # Features
    y = df_encoded["DRUH_POVR"]  # Target

    # Convert features to TensorFlow tensors
    X_tensor = tf.convert_to_tensor(X.to_numpy(), dtype=tf.float32)

    # Use TensorFlow StringLookup for label encoding
    label_lookup = tf.keras.layers.StringLookup(output_mode='int', vocabulary=tf.constant(y.unique()))
    y_tensor = label_lookup(y) - 1  # Adjust labels to start from 0

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

    # Shuffle and split the dataset
    dataset_size = len(y_tensor)
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)

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

    return train_dataset, val_dataset, test_dataset, len(label_lookup.get_vocabulary()) - 1


def main():
    flare.init()

    sys_info = flare.system_info()
    print(f"System info: {sys_info}", flush=True)

    # Load data and determine input/output shapes
    train_dataset, val_dataset, test_dataset, num_classes = load_and_preprocess_data()

    # Initialize model
    model = TFNet(input_shape=(train_dataset.element_spec[0].shape[-1],), num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()

    while flare.is_running():
        input_model = flare.receive()
        print(f"Current round: {input_model.current_round}")

        # Update model weights with the received model
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)

        # Evaluate the global model
        _, test_global_acc = model.evaluate(test_dataset, verbose=2)
        print(
            f"Global model accuracy on round {input_model.current_round}: {test_global_acc * 100:.2f}%"
        )

        # Train locally
        model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=1)

        print("Finished Training")

        # Save weights for debugging or inspection
        model.save_weights(WEIGHTS_PATH)

        # Prepare and send updated model
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers},
            params_type="FULL",
            metrics={"accuracy": test_global_acc},
            current_round=input_model.current_round,
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
