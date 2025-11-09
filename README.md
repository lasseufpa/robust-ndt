# Towards a Robust Transport Network With a Self-adaptive Network Digital Twin

## :bulb: Introduction

![figure_1](https://github.com/lasseufpa/robust-ndt/blob/main/figures/figure_1.jpg?raw=true)

The ability of the network digital twin (NDT) to remain aware of changes in its physical counterpart, known as the physical twin (PTwin), is a fundamental condition to enable timely synchronization, also referred to as twinning. In this way, considering a transport network, a key requirement is to handle unexpected traffic variability and dynamically adapt to maintain optimal performance in the associated virtual model, known as the virtual twin (VTwin).In this context, we propose a self-adaptive implementation of a novel NDT architecture designed to provide accurate delay predictions, even under fluctuating traffic conditions. This architecture addresses an essential challenge, underexplored in the literature: improving the resilience of data-driven NDT platforms against traffic variability and improving synchronization between the VTwin and its physical counterpart. 

## :open_file_folder: Repository directory structure
```bash
├── data_management -> directory with different scripts to process raw data and transform into tensorflow-like datasets
│   ├── labeled_database -> Virtual twin directory to store labeled data for retraining process
│   ├── traffic_database -> Virtual twin directory to store data used in the prediction process
│   └── weights_database -> Virtual twin directory to store model weights after each training process
├── ndt
│   ├── alone_training -> directory with standalone training script and VTwin declaration model
│   └── sync
│       ├── delay_database -> directory with database with delay as the target metric
│       ├── database_for_app -> directory with database with packet delay budget
│       ├── jitter_database -> directory with database with jitter as the target metric
│       ├── misc -> directory with plot scripts
│       └── results -> directory with NDT operation across different realizations
└── physical_twin -> directory with all scripts to generate traffin in the physical twin 
│   └── topologies -> directory with different transport network topologies in GML format

```

## :writing_hand: Getting started
To install the python environment used to conduct the experiments, use the following command:
### :package: Install conda environment
```bash
conda env create -f environment.yml
```

Once the environment installed, you need to instatiate a container with the SDN controller (ONOS). Noteworthy that this is necessary only if you would like to generate traffic datasets. So, considering the current directory `physical_twin`, you can run the following command:

### Instantiate ONOS docker container
```
sudo docker compose up -d
```

You can also access the ONOS graphical interface in your browser, using the following URL:

```
http://<your-ip>:8181/onos/ui/login.html
```

Where, `<your-ip>` is the machine IP where ONOS is running. 

Furthermore, the default credentials to login into the SDN controller GUI is:

```
user: onos
password: rocks
```


## :test_tube: Data generation
The data generation is process that starts with the generation of the raw dataset using the D-ITG simulator using `generate_traffic.py`. However, this data generated, at this stage, is not suitable for training a deep learning model (VTwin). So, we need to extract all features available using `generate_metrics.py` script, to finally generate Tensorflow-like data, using `generate_data.py` script, to be used by the VTwin model. 

### Generate raw network traffic data with D-ITG
Flags:

`--topo-filepath`: Path to `.gml` topology.

`--id`: Simple simulation index.

`--pattern`: Type of traffic pattern which can (`exp` | `normal` | `poisson` | `pareto` | `gamma` | `burst`).

`--pkt-size`: Packet size in bytes.

`--duration`: Total simulation time in seconds.

Example of use, considering the current directory `physical_twin`
```bash
generate_traffic.py --id 1 --topo-filepath topologies/nsfnet_14.gml --pattern exp --pkt-size 512 --duration 20
```

The command above will generate a network traffic in the NSFNet topology, with packet rate and packet size following an exponential distribution, and a total duration of 20 seconds.

:warning: The dataset generated from this command can consume to much space. Ensure you have enough space for large simulations.

:information_source: If you prefer to use the raw dataset generated for this work, you can access it by this [link](https://nextcloud.lasseufpa.org/s/fotcwk6NGsELBnr).

The table below describe the characteristics of the datasets generated.

| # | Traffic pattern  | Topology      | Raw dataset id      | Number of flows (T) |
|---|------------------|---------------|---------------------|---------------------|
|   |                  | 5G-Crosshaul  |experiment_100       | 121,400             |
| 1 |Exponential       | Germany       |experiment_200       | 129,600             |
|   |                  | PASSION       |experiment_300       | 129,600             |
|   |                  | 5G-Crosshaul  |experiment_101       | 234,501             |
| 2 | Poisson          | Germany       |experiment_201       | 257,401             |
|   |                  | PASSION       |experiment_301       | 243,001             |
|   |                  | 5G-Crosshaul  |experiment_102       | 347,002             |
| 3 | Uniform          | Germany       |experiment_202       | 378,802             |
|   |                  | PASSION       |experiment_302       | 355,002             |
|   |                  | 5G-Crosshaul  |experiment_301       | 447,803             |
| 4 |Deterministic     | Germany       |experiment_302       | 440,003             |
|   |                  | PASSION       |experiment_304       | 478,203             |

### Generate metrics from network traffic dataset
Once a network traffic is generated by the D-ITG simulator, you need to process the logs files generated. To do so, you can use the script `generate_metrics.py` for this task. This script will retrieve metrics such as `average traffic rate`, `delay`, `jitter` and `packet loss`.

Flags:

`--dir`: Path to network traffic log directory.


```bash
sudo generate_metrics.py --dir experiments_100
```

### Generate data to VTwin model
Finally, once the metrics were processed, you can generate data, typically input-output format to be used by the VTwin (GNN) model.

Flags:

`--input-dir`: Path to processed metrics.

`--output-dir`: Path to output dataset.

```bash
python3 generate_data.py --input-dir [path-to-processed-dir] --output-dir [path-to-output-dir]
```

## :gear: NDT operation

To run a transport network NDT, you can use standlone script from `alone_training`, which there is no synchronization elements, or the scripts in `sync` directory with synchronization elements.

### NDT with synchronization
To generate the results and run the NDT for transport network with synchronization, you can run the `ndt_synchronization.py` script. To customize this execution, you can use the following flags:

`--topology`: Topology to be adopted in the NDT simulations.

`--database`: Path to retraining data.

`--realization`: Number of independent realization to used in the simulations.

`--sync`: Flag to enable NDT synchronization.

`--target`: Type of QoS to be predicted.

Example of use, considering the current directory `ndt/sync`:

```bash
python3 ndt_synchronization.py --topology topologies/5g_crosshaul --dir delay_database --realization 10 --sync --target delay 
```

In the above example, the transport network NDT will running considering the 5G-Crosshaul, the QoS metric to be predicted will be the per-flow delay, and 10 traininig process realization will be taken.

### Application for SLA monitoring
You can run the SLA monitoring application using the script `ndt_sync_w_app.py`. This application can running in different topologies, with different number of realization and with or without synchronization. To configure this, you can the following flags:

`--topology`: Topology to be adopted in the NDT simulations.

`--database`: Path to retraining data.

`--realization`: Number of independent realization to used in the simulations.

`--sync`: Flag to enable NDT synchronization.

`--target`: Type of QoS to be predicted.

Example of use, considering the current directory `ndt/sync`:

```bash
python3 ndt_sync_w_app.py --topology 5g_crosshaul --dir delay_database --realization 1 --sync 
```
In this case, the SLA monitoring consider only the predicted per-flow delay to classify whether a flow is in compliance with its SLA. 

## :bar_chart: Result plots

Several scripts in the `misc` directory can be used to reproduce the results from the paper:

- `concept_drift_op_plot.py`: Visualizes concept drift detection across topologies (5G-Crosshaul, Germany, PASSION).
- `get_nmse_metrics.py`: Computes NMSE metrics across all NDT realizations.
- `histogram_plot.py`: Displays topology characteristics (capacity and propagation delay).
- `single_plot.py`: Plots NMSE performance for a single topology.
- `multiple_plots.py`: Compares NMSE performance across two topologies (with/without synchronization).
- `sla_violations_plot.py`: Visualizes SLA monitoring results with and without synchronization.
- `training_time_plot.py`: Shows average retraining time per topology after different concept drift events.
- `window_size_plot.py`: Evaluates detected concept drifts across different window sizes (hyperparameter).

## :information_source: Credits

If you benefit from this work, please cite on your publications using:
```
@misc{modesto2025,
      title={Towards a Robust Transport Network With Self-adaptive Network Digital Twin}, 
      author={Cláudio Modesto and João Borges and Cleverson Nahum and Lucas Matni and Cristiano Bonato Both and Kleber Cardoso and Glauco Gonçalves and Ilan Correa and Silvia Lins and Andrey Silva and Aldebaro Klautau},
      year={2025},
      eprint={2507.20971},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2507.20971}, 
}
```