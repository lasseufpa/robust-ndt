'''
Script for training GNN model
'''

import os
import random
from typing import List, Optional, Union, Tuple, Dict, Any
import tensorflow as tf
import numpy as np

# Run eagerly-> Turn true for debugging only
RUN_EAGERLY = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.run_functions_eagerly(RUN_EAGERLY)

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


def get_default_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Returns the default callbacks for the training of the models
    (EarlyStopping and ReduceLROnPlateau callbacks)
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.0002,
            start_from_epoch=4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
            min_delta=0.001,
        ),
    ]


def get_default_hyperparams() -> Dict[str, Any]:
    """Returns the default hyperparameters for the training of the models. That is
    - Adam optimizer with lr=0.001
    - MeanAbsolutePercentageError loss
    - EarlyStopping and ReduceLROnPlateau callbacks
    """
    return {
        "optimizer": tf.keras.optimizers.AdamW(learning_rate=0.001),
        "loss": tf.keras.losses.MeanAbsolutePercentageError(),
        "metrics": ['MeanAbsolutePercentageError'],
        "additional_callbacks": get_default_callbacks(),
        "epochs": 50,
    }


def get_mean_std_dict(
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the mean-std for different parameters of a dataset. 
    Later used by the models for the z-score normalization.

    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the mean-std normalization from.

    params : List[str]
        List of strings indicating the parameters to extract the features from.

    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not included.
        By default None.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the z-score normalization.
        The first value is the min value of the parameter, and the second is 1 / (max - min).
    """

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        mean_val = np.mean(param_list, axis=0)
        std_val = np.std(param_list, axis=0)

        if all(std_val) == 0:
            scores[param] = [mean_val, std_val]
        else:
            scores[param] = [mean_val, 1/(std_val)]

    return scores


def train_and_evaluate(
    ds_path: Union[str, Tuple[str, str]],
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics: List[tf.keras.metrics.Metric],
    additional_callbacks: List[tf.keras.callbacks.Callback],
    epochs: int = 100,
    ckpt_path: Optional[str] = None,
    tensorboard_path: Optional[str] = None,
    final_eval: bool = True,
) -> Tuple[tf.keras.Model, Union[float, np.ndarray, None]]:
    """
    Train the given model with the given dataset, using the provided parameters
    Besides for defining the hyperparameters, refer to get_default_hyperparams()

    Parameters
    ----------
    ds_path : str
        Path to the dataset. Datasets are expected to be in tf.data.Dataset format, and to 
        be compressed with GZIP. If ds_path is a string, then it used as the path to both the
        training and validation dataset. If so, it is expected that the training and
        validation datasets are located in "{ds_path}/training" and "{ds_path}/validation" respectively.
        If ds_path is a tuple of two strings, then the first string is used as the path to the training dataset,
        and the second string is used as the path to the validation dataset.

    model : tf.keras.Model
        Instance of the model to train. Besides being a tf.keras.Model, it should have the same constructor and the name parameter
        as the models in the models module.

    optimizer : tf.keras.Optimizer
        Optimizer used by the training process

    loss : tf.keras.losses.Loss
        Loss function to be used by the process

    metrics : List[tf.keras.metrics.Metric]
        List of additional metrics to consider during training

    additional_callbacks : List[tf.keras.callbacks.Callback], optional
        List containing tensorflow callback functions to be added to the training process.
        A callback to generate tensorboard and checkpoint files at each epoch is already added.

    epochs : int, optional
        Number of epochs of in the training process, by default 100

    ckpt_path : Optional[str], optional
        Path where to store the training checkpints, by default "{repository root}/ckpt/{model name}"

    tensorboard_path : Optional[str], optional
        Path where to store tensorboard logs, by default "{repository root}/tensorboard/{model name}"

    final_eval : bool, optional
        If True, the model is evaluated on the validation dataset one last time after training, by default True

    Returns
    -------
    Tuple[tf.keras.Model, Union[float, np.ndarray, None]]
        Instance of the trained model, and the result of its evaluation
    """

    # Reset tf state
    _reset_seeds()
    # Check epoch number is valid
    assert epochs > 0, "Epochs must be greater than 0"
    # Load ds
    if isinstance(ds_path, str):
        ds_train = tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP")
        ds_val = tf.data.Dataset.load(f"{ds_path}/validation", compression="GZIP")
        ds_test = tf.data.Dataset.load(f"{ds_path}/testing", compression="GZIP")

    # Checkpoint path
    if ckpt_path is None:
        ckpt_path = f"ckpt/{model.name}"
    # Tensorboard path
    if tensorboard_path is None:
        tensorboard_path = f"tensorboard/{model.name}"

    # Apply z-score normalization
    model.set_mean_std_scores(get_mean_std_dict(ds_train, model.mean_std_scores_fields))

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=RUN_EAGERLY,
    )
    
    #Create callbacks
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_path, "final_weight"),
        verbose=1,
        mode="min",
        save_best_only=False,
        save_weights_only=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_path, histogram_freq=1
    )

    nan_callback = tf.keras.callbacks.TerminateOnNaN(),

    # Train model
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        verbose=1,
        callbacks=[ckpt_callback, tensorboard_callback, nan_callback] + additional_callbacks,
        use_multiprocessing=True,
    )

    return model, model.evaluate(ds_test)


if __name__ == "__main__":
    import argparse
    import std_delay_model

    parser = argparse.ArgumentParser(
        description="Train a model for flow delay prediction"
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--ds-train", type=str, required=True)

    args = parser.parse_args()

    # Check the scenario
    ds_path = args.ds_train
    ckpt_path = args.ckpt_path
    Model = std_delay_model.VirtualTwin

    # code for simple training/validation
    _reset_seeds()
    trained_model, evaluation = train_and_evaluate(
        os.path.join(ds_path),
        Model(),
        **get_default_hyperparams(),
        ckpt_path=ckpt_path
    )

    print(evaluation)
