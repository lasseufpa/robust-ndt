"""
Virtual twin with drift detection main script
Created by Cláudio Modesto
"""

import os
import random
from time import sleep
import pathlib
import subprocess
import numpy as np
from river import drift
from matplotlib import pyplot as plt
from std_train import get_mean_std_dict, train_and_evaluate, get_default_hyperparams
import tensorflow as tf
import std_delay_model

model_version = 0
async_running = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main_loop():
    """
    Main loop traffic generation function
    """
    global model_version
    global async_running
    admin_permission = True

    convey_time = 0 # just to accelerate the simulation
    threshold = -5 # NMSE performance threshold
    alarm_limit = 3 # limit of samples windows with a NMSE greater than "alarm_limit"
    false_alarm = 0 # number of false positive 
    root_data_dir = "/home/claudio/papers/2025-cn-netwins/traffic_generator/ditg"
    dataset_name = "experiment_10"

    model_version = 0 # model version id
    training_data = [f"{root_data_dir}/labeled_data/{dataset_name}0_cv",
                    f"{root_data_dir}/labeled_data/{dataset_name}1_cv",
                    f"{root_data_dir}/labeled_data/{dataset_name}2_cv",
                    f"{root_data_dir}/labeled_data/{dataset_name}4_cv"]

    model_weights_dir = "weights/model_version"

    if not os.path.isfile(f"{model_weights_dir}_{model_version}/final_weight.index"):
        untrained_model = _load_untrained_model()
        training_vtwin(training_data[model_version], untrained_model)

    stream_data_dir = [f"{root_data_dir}/labeled_data/{dataset_name}0_cv/testing",
                        f"{root_data_dir}/labeled_data/{dataset_name}1_cv/testing",
                        f"{root_data_dir}/labeled_data/{dataset_name}2_cv/testing",
                        f"{root_data_dir}/labeled_data/{dataset_name}4_cv/testing"]

    weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"
    last_weight_id = os.path.getmtime(weight_filename)
    stream_data = tf.data.Dataset.load(f"{stream_data_dir[0]}", compression="GZIP")

    for stream_dir in stream_data_dir[1:]:
        new_data = tf.data.Dataset.load(stream_dir,
                                                compression="GZIP")
        stream_data = stream_data.concatenate(new_data)

    trained_model = _load_trained_model(training_data[model_version],
                                        f"{model_weights_dir}_{model_version}/final_weight")
    nmses, indexes, points = [], [], []
    flow_id = 0

    model_training = None
    window_index = 0
    model_updated = []
    drift_detected = []
    for _, (sample_features, labels) in enumerate(stream_data):
        stream_data_sample = (sample_features, labels)
        weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"

        # update virtual twin model
        if (os.path.isfile(weight_filename) and \
            os.path.getmtime(weight_filename) > last_weight_id \
                                        and admin_permission \
                                        and not async_running):
            print("New model")
            model_updated.append(window_index)
            last_weight_id = os.path.getmtime(weight_filename)
            print(f"Model version {model_version} in production")
            trained_model = _load_trained_model(training_data[model_version],
                                    f"{model_weights_dir}_{model_version}/final_weight")

        nmse = predicting_vtwin(trained_model, stream_data_sample)
        nmses.append(nmse)

        plt.plot([max(x, -50) for x in nmses])
        plt.xlabel("Window index")
        plt.ylabel("NMSE (dB)")
        plt.savefig("error_metric.pdf")
        plt.close()

        for _ in sample_features["flow_traffic"].numpy():
            print(flow_id)
            if nmse > threshold:
                false_alarm += 1
            else:
                false_alarm = 0
            if false_alarm > alarm_limit and admin_permission and not async_running:
                drift_detected.append(window_index)
                false_alarm = 0
                if model_version + 2 > len(training_data):
                    break
                print("DRIFT detected")
                print("Retraing the model")
                convey_time = 0.8
                async_running = True
                model_version += 1
                model_training = subprocess.Popen(["python3", "std_train.py",
                                "--ds-train", f"{training_data[model_version]}",
                                "--ckpt-path", f"{model_weights_dir}_{model_version}"])
            flow_id += 1
            sleep(convey_time)

        if async_running and model_training.poll() is not None:
            print("Retraining is over")
            convey_time = 0
            async_running = False
        window_index += 1
    if model_training is not None:
        model_training.kill()
    print("Saving Error Metrics")
    with open(f"mapes_with_admin_{admin_permission}.npz", "wb") as f:
        np.savez(f, np.array(nmses), drift_detected, model_updated)

def _load_untrained_model():
    model = std_delay_model.VirtualTwin

    return model


def _load_trained_model(training_data, model_weights_file):
    """
    Load a trained GNN model
    """
    model = std_delay_model.VirtualTwin()

    model.load_weights(f"{model_weights_file}")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        metrics=tf.keras.metrics.MeanAbsolutePercentageError()
    )

    model.set_mean_std_scores(
        get_mean_std_dict(
            tf.data.Dataset.load(f"{training_data}/training",
            compression="GZIP"),
            model.mean_std_scores_fields,
        )
    )

    return model


def predicting_vtwin(trained_model, stream_data):
    """
    function to prediction per-flow delay using
    a trained GNN model
    """
    # obtain the prediction as numpy array, and flatten
    predicted_delay = trained_model(stream_data[0]).numpy().reshape((-1,))
    ground_truth_delay = stream_data[1].numpy()
    err_metric = np.mean((ground_truth_delay - predicted_delay)**2)/np.mean(ground_truth_delay**2)
    err_metric = 10*np.log10(err_metric)

    return err_metric


def training_vtwin(training_data, model):
    """
    function to training GNN model
    """
    global async_running
    global model_version

    model_weights_dir = f"weights/model_version_{model_version}/"
    pathlib.Path(model_weights_dir).mkdir(parents=True, exist_ok=True)
    _reset_seeds()
    train_and_evaluate(
        os.path.join(training_data),
        model(),
        **get_default_hyperparams(),
        ckpt_path=model_weights_dir
    )


def _reset_seeds(seed: int = 42) -> None:
    """Reset rng seeds, and also indicate tf if to run eagerly or not

    Parameters
    ----------
    seed : int, optional
        Seed for rngs, by default 42
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main_loop()
    