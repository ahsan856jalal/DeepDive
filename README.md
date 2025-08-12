# DeepDive
Fish length estimation in unconstrained underwater environment using monocular imagery

Codes folder contains all instructions and files to get results on Ozfish dataset
```
conda create --name DeepDive
conda activate DeepDive
```

## Instructions for setup 

Install DepthAnything-V2

```
git clone https://github.com/DepthAnything/Depth-Anything-V2

cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
cd ..
pip install -r requirements.txt  
```
Download Vitb model from the link and copy to checkpoints folder

```
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

cp depth_anything_v2_vitb.pth checkpoints/
```

Install ultralytics
```
pip install ultralytics

```
## Download all model files and data folders from Google drive link to data folder
```
mkdir data

cd data
```
[Google Drive link](https://drive.google.com/drive/folders/1J23uHRA1eduPEjRq5i_9Bolgmq_gvY16?usp=drive_link)


# Length Estimation and Metrics calculations
Make sure Length estimation files are run from Depth-Anything V2 main dir
## For Ozfish
```
python length_estimation_ozfish.py
python calculate_eval_metrics.py --filepath ../data/estimated_lengths_ozfish.csv

```
## For SBT
```
python length_estimation_sbt.py
python calculate_eval_metrics.py --filepath ../data/estimated_lengths_sbt.csv

```



