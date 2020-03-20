# E-PGN 
**Tensorflow** implementation of 'Episode-based Prototype Generating Network for Zero-Shot Learning'

![image](https://github.com/yunlongyu/EPGN/blob/master/img/one_episode.png)

## Preparation

- **Prerequisites**
    - Tensorflow (r1.4 - r1.12 should work fine)
    - Python 2.7 with matplotlib, numpy and scipy
    
- **Datasets**
   - download the model from [Google Drive](https://drive.google.com/open?id=1FtIe_SC70jTy2TKp3aKokFozLMQie579) or [Baidu Cloud (bphc)](https://pan.baidu.com/s/1F0GSYAhCPBwVEL4kqqaLww), and unzip the files to ***./data/***,

## Training & Test

Exemplar commands are listed here for AwA1 dataset.
- You can 
    ```console
    sh ./run_script/1awa.sh
    ```
- or
   ```console
    python ../scripts/1awa.py --att_dim 85 --hid_dim 1800 --mid_dim 1200 --cla_num 50 --tr_cla_num 40 --selected_cla_num 10 --lr 5e-5 --epoch 30 --episode 50 --inner_loop 100 --batch_size 100 --dropout --manualSeed 4198 
   ```


