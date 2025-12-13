"""
Script to process raw traffic data and transform into .txt files
"""
import glob
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir", "-d", help="Path to network traffic datasets logs.", 
    type=str, required=True
)

args = parser.parse_args()

root_dir = args.dir
for exp_file in glob.glob(f'{root_dir}/*_traffic_results_rx.log'):
    print(exp_file)
    exp_conn_id = int(exp_file.split('/')[-1].split('_')[0])
    if os.path.isfile(f'{root_dir}/{exp_conn_id}_metric_results.txt'):
        continue
    os.system(f'ITGDec {exp_file} -c 1000 {root_dir}/{exp_conn_id}_metric_results.txt')
