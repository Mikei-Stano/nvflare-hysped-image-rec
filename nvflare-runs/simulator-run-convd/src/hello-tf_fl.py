import os
import uuid
import tensorflow as tf
import pandas as pd
import nvflare.client as flare
import wandb
from codecarbon import EmissionsTracker
from wandb.integration.keras import WandbCallback
from tf_net import TFNet

# Paths
WEIGHTS_PATH = "./tf_model.weights.h5"
DATASET_PATH = "/workspace/smaller_dataset.csv"  # Update this path if needed

# Ensure NVFL_SITE_NAME is set correctly
client_id = os.environ.get("NVFL_SITE_NAME", "unknown_client")
if client_id == "unknown_client":
    print("⚠ Warning: NVFL_SITE_NAME is not set! Using 'unknown_client'")

# Create unique run ID (stored per client for potential resumption)
wandb_id_file = f"./wandb_id_{client_id}.txt"
wandb_id = str(uuid.uuid4())

if os.path.exists(wandb_id_file):
    with open(wandb_id_file, "r") as f:
        wandb_id = f.read().strip()
    print(f"🔄 Resuming previous W&B run with ID: {wandb_id}")
else:
    with open(wandb_id_file, "w") as f:
        f.write(wandb_id)

# Initialize Weights & Biases
wandb.login(key="2a6d75e62c637b3c3f6f727ad4ebe3e85efe85be")
wandb.init(
    project="nvflare-fl-training",
    name=f"client_{client_id}",
    id=wandb_id,
    config={
        "batch_size": 32,
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "architecture": "Conv1D",
    },
    resume="allow",
    reinit=True
)


def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    df_encoded = df.drop(columns=[df.columns[0]])
    X = df_encoded.drop(columns=["DRUH_POVR", "NAZ_LOKALI"])
    y = df_encoded["DRUH_POVR"]

    X_tensor = tf.convert_to_tensor(X.to_numpy(), dtype=tf.float32)
    X_tensor = tf.expand_dims(X_tensor, axis=-1)

    label_lookup = tf.keras.layers.StringLookup(output_mode='int', vocabulary=tf.constant(y.unique()))
    y_tensor = label_lookup(y) - 1  # Adjust labels to start at 0

    num_classes = len(label_lookup.get_vocabulary()) - 1
    y_one_hot = tf.one_hot(y_tensor, depth=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_one_hot)).shuffle(len(y), seed=42)

    # Train, Validation, Test Split
    train_size, val_size = int(0.7 * len(y)), int(0.15 * len(y))
    train_dataset = dataset.take(train_size).batch(32)
    val_dataset = dataset.skip(train_size).take(val_size).batch(32)
    test_dataset = dataset.skip(train_size + val_size).batch(32)

    return train_dataset, val_dataset, test_dataset, num_classes


def main():
    flare.init()
    print(f"🖥 System info: {flare.system_info()}", flush=True)

    train_dataset, val_dataset, test_dataset, num_classes = load_and_preprocess_data()
    input_shape = train_dataset.element_spec[0].shape[1:]

    # Initialize Model
    model = TFNet(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.summary()

    # Initialize CodeCarbon for electricity tracking (Start only ONCE)
    tracker = EmissionsTracker(log_level="critical", measure_power_secs=1)
    tracker.start()

    while flare.is_running():
        input_model = flare.receive()
        print(f"🔄 Current round: {input_model.current_round}")

        for k, v in input_model.params.items():
            try:
                model.get_layer(name=k).set_weights(v)
            except ValueError:
                print(f"⚠ Layer {k} not found. Skipping update.")

        # Evaluate before training
        loss, test_global_acc = model.evaluate(test_dataset, verbose=2)
        print(f"📊 Global model accuracy: {test_global_acc * 100:.2f}%")

        wandb.log({
            f"{client_id}/global_model_accuracy": test_global_acc,
            "round": input_model.current_round
        })

        # Train Locally
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,  # One local epoch per round
            verbose=1,
            callbacks=[WandbCallback()]
        )

        # Stop tracking electricity after training
        co2_emissions_kg = tracker.stop()
        energy_consumed_kwh = getattr(tracker, "_last_emissions_data", {}).get("energy_consumed", 0)  # Fix energy tracking

        # Log Training Stats
        wandb.log({
            f"{client_id}/train_loss": history.history["loss"][-1],
            f"{client_id}/val_loss": history.history["val_loss"][-1],
            f"{client_id}/train_accuracy": history.history["categorical_accuracy"][-1],
            f"{client_id}/val_accuracy": history.history["val_categorical_accuracy"][-1],
            f"{client_id}/electricity_consumption_kWh": energy_consumed_kwh,  # ✅ Corrected
            f"{client_id}/co2_emissions_kg": co2_emissions_kg,  # ✅ Corrected
            "round": input_model.current_round
        })

        # Save model weights
        model.save_weights(WEIGHTS_PATH)

        # Send Updated Model Parameters
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers if layer.weights},
            params_type="FULL",
            metrics={"categorical_accuracy": test_global_acc},
            current_round=input_model.current_round,
        )
        flare.send(output_model)

    # Stop CodeCarbon & Finish W&B Session
    tracker.stop()
    wandb.finish()


if __name__ == "__main__":
    main()
