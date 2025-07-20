import glob
import os

root_dir = 'logs/experiment_400'
for exp_file in glob.glob(f'{root_dir}/*_traffic_results_rx.log'):
    print(exp_file)
    exp_conn_id = int(exp_file.split('/')[-1].split('_')[0])
    if os.path.isfile(f'{root_dir}/{exp_conn_id}_metric_results.txt'):
        continue
    os.system(f'ITGDec {exp_file} -c 1000 {root_dir}/{exp_conn_id}_metric_results.txt')
