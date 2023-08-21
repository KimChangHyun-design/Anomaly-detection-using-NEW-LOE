# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from config.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simu_num', dest='simu_num', default='1')
    parser.add_argument('--config-file', dest='config_file', default='config_cifar10.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='cifar10')
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--anomaly_rate', type=float, default=0.1)
    parser.add_argument('--model_configuration.train_method ',dest='train_method', default='loe_soft_semi')
    parser.add_argument('--query_num', type=int, default=0) # for active anomaly detection
    return parser.parse_args()

def EndtoEnd_Experiments(simu_num,config_file, dataset_name,contamination,anomaly_rate,query_num,train_method):

    model_configurations = Grid(config_file, dataset_name)
    model_configurations[0]['train_method'] =train_method
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    #result_folder = model_configuration.result_folder+model_configuration.exp_name
    #exp_path = os.path.join(result_folder,f'{contamination}_{model_configuration.train_method}')
    
    result_folder = model_configuration.result_folder+model_configuration.exp_name
    exp_path = os.path.join(result_folder,f'Simulation_model_{simu_num}_c_{contamination}_a_{anomaly_rate}_{train_method}')
    
    risk_assesser = KVariantEval(dataset, exp_path, model_configurations,contamination,query_num,anomaly_rate,simu_num)

    risk_assesser.risk_assessment(runExperiment)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file
    EndtoEnd_Experiments(args.simu_num,config_file, args.dataset_name,args.contamination,args.anomaly_rate,args.query_num,args.train_method)
