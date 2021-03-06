# FusionCNN_hurricanes

### Public codes for fusing CNN networks (for hurricane forecasting)
see 'Tropical Cyclone Track Forecasting using Fused Deep Learning from Aligned Reanalysis Data', 2020.
Sophie Giffard-Roisin, Mo Yang, Guillaume Charpiat, Balázs Kégl, Claire Monteleoni, Frontiers in Big Data - Data-driven Climate Sciences https://arxiv.org/abs/1910.10566

This code is data-specific. However, the blocks could be easily re-used for other problems.

### Also included: data collection/processing functions.
Processing and collection of ERAInterm data and IBTRACS data. See the scripts_data_collect_process/
and the DataProcessing/ module. Be aware that ERAInterm reanalysis is already outdated... but the codes can be re-used for ERA5 data, also freely avaible online.

## To keep in mind:
The data to run this code is not directly available (because it is large and already outdated...!).

   1) you can use the DataProcessing module functions to help you for retrieving/processing the track and/or reanalysis data avalailable online. Please look at the Discussion section of this paper (https://arxiv.org/abs/1910.10566) in order to make the good data processing choices, such as the new ERA5 reanalysis. See the databases: https://apps.ecmwf.int/datasets/ and https://www.ncdc.noaa.gov/ibtracs

   2) you can use the fusion code for another task by changing the dataloader, loss functions...

   3) you can ask me for more information (sophie.giffard at univ-grenoble-alpes.fr)

## How to run it:
1) launch the 3 separate data-stream neural network trainings: 
    - script_train_0D_model.py for the simple 0D neural network
    - script_train_single_models_2D.py for the 2 CNNs (separate trainings)

2) launch the fusion training by loading first the 3 trained models saved.
    - script_launch_fusion.py, don't forget to add the save result part.
    

## example of usage:

    script_train_0D_model.py displacement --lr=1e-3 --weight-decay=0.01 --epochs=200 --num_tracks=2 --hours=24 --save_fig_name='0D_model'

    script_train_single_models_2D.py displacement --lr_0=1e-3 --lr_1=1e-5 --l_2=1e-7 --weight-decay=0.01 --epochs_0=50 --epochs_1=500 --epochs_2=10  --hours=24 --save_fig_name='uv_model'

    script_launch_fusion.py displacement --lr=1e-3 --weight-decay=0.01 --epochs_freeze=50 --epochs_final=500 --hours=24 --save_fig_name='uv0d_fusion'



![Alt text](img/fusion_network.png?raw=true "fusion network")
