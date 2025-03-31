# NETwins Journal Paper

## Getting started
### Install conda environment
```bash
conda env create -f environment.yml
```

### Instantiate ONOS docker container
```
sudo docker compose up -d
```

### Generate traffic with iPerf
Flags:

`--topo-filepath`: Path to gml topology.

`id`: Simple simulation index
```
sudo mn -c && sudo $(which python) generate_traffic.py --id 1 --topo-filepath topology/nsfnet_14.gml
```