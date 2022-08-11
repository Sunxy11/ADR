## ADR
Tensorflow implementation of our unsupervised domain adaptation cardiac segmentation framework. 
![image](https://github.com/Sunxy11/ADR/blob/main/picture/Overview.png)
## Requirements
- Tensorflow 1.14.0
- python 3.6
- CPU or GPU + CUDA CuDNN

## Dataset
* The dataset used is the same as SIFA. The training data can be downloaded [here](https://drive.google.com/file/d/1m9NSHirHx30S8jvN0kB-vkd7LL0oWCq3/view). The testing CT data can be downloaded [here](https://drive.google.com/file/d/1SJM3RluT0wbR9ud_kZtZvCY0dR9tGq5V/view). The testing MR data can be downloaded [here](https://drive.google.com/file/d/1Bm2uU4hQmn5L3GwXz6I0vuCN3YVMEc8S/view?usp=sharing).
* Put 'tfrecords' training data and 'npz' test data of two domains into corresponding folders under `./data` accordingly.
* Run './create_datalist.py' to generate the datalists containing the path of each data.
* Run './convertToNpz.py' to convert the 'nii.gz' file to 'npz' file.

## Train
* Run './readCkpt.py' to get the initial model of coarse alignment in our model.
* Run './main.py' to start the training process

## Evaluate
* Our trained models can be downloaded from [here](https://pan.baidu.com/s/1rQWs2AkqT-uZbbK5O3hk3w?pwd=9u51).
* Run './evaluate.py' to start the evaluation.

## Acknowledgement
This code is heavily borrowed from [SIFA](https://github.com/cchen-cc/SIFA), partly from [DGNet](https://github.com/vios-s/DGNet).

## Citation
If you use this code for your research, please cite our papers.
