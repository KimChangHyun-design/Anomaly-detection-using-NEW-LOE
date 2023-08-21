# Enhanced Deep Anomaly Detection In Contaminated Datasets Using Semi-Supervised Learning


## Reproduce the Results


Please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --simu_num $0--config-file $1 --dataset-name $2 --contamination $3 --anomaly_rate $4 --model_configuration.train_method $5
```


**simu_num**
* 1;2;3;4;5;6

**config-file:** 
* config_simulation_ntl_plus_soft.yml; config_simulation_ntl_plus_hard.yml; 

**dataset-name:** 
* simulation_data_ntl_plus;

**contamination:** 
* The ground-truth contamination ratio of the dataset. The default ratio is 0.1.

**anomaly_rate:** 
* The known anoamly ratio of the dataset. The default ratio is 0.1.

**model_configuration.train_method** 
*loe_hard; loe_soft; loe_soft_semi; loe_hard_semi; gt; refine; blind

## How to Use
1. When using your own data, please put your data files under [DATA](DATA).

2. Create a config file which contains your hyper-parameters under [config_files](config_files).  

3. Add your data loader to the function ''load_data'' in the [loader/LoadData.py](loader/LoadData.py).
* The shape is (batch size, feature dim).

##Colab reproduce
#from google.colab import drive
#drive.mount('/content/drive')
#import os
#os.getcwd()
#os.chdir("/content/drive/MyDrive/LOE_semi")
#!python Launch_Exps.py --simu_num 1 --config-file config_simulation_ntl_plus_soft.yml --dataset-name simulation_data_ntl_plus --contamination 0.1 --anomaly_rate 0.2 --model_configuration.train_method loe_hard
