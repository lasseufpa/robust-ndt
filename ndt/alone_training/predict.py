import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import std_delay_model
from std_train import get_mean_std_dict


root_data_dir = '../ndt/weights/model_version_2'
trained_model = std_delay_model.VirtualTwin()

trained_model.load_weights(f'{root_data_dir}/final_weight')

# Compile the model
trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanAbsolutePercentageError(),
    metrics=tf.keras.metrics.MeanAbsolutePercentageError()
)

dataset_path = '/home/claudio/papers/2025-cn-netwins/traffic_generator/ditg/labeled_data/experiment_204_cv'
trained_model.set_mean_std_scores(
    get_mean_std_dict(
        tf.data.Dataset.load(f'{dataset_path}/training',
        compression='GZIP'),
        trained_model.mean_std_scores_fields,
    )
)

stream_data = tf.data.Dataset.load(f'{dataset_path}/testing', compression="GZIP")

aes = []
for _, (sample_features, labels) in enumerate(stream_data):
    stream_data_sample = (sample_features, labels)
    # obtain the prediction as numpy array, and flatten
    predicted_delay = trained_model(stream_data_sample[0]).numpy().reshape((-1,))
    ground_truth_delay = stream_data_sample[1].numpy()
    #mape = np.mean(np.abs((ground_truth_delay - predicted_delay)/ground_truth_delay)) * 100
    ae = np.abs((ground_truth_delay - predicted_delay))
    aes.extend(ae)

plt.plot(aes)
plt.savefig('mape.pdf')
