# instructions for using 

Install DepthAnything-V2

```
git clone https://github.com/DepthAnything/Depth-Anything-V2

cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
```

Install ultralytics
```
pip install ultralytics

```
## Download all model files and data folders from Google drive link to codes folder
```
cd codes
```
[Google Drive link](https://drive.google.com/drive/folders/1J23uHRA1eduPEjRq5i_9Bolgmq_gvY16?usp=drive_link)


# To run Evaluation code and get accuracy metrices 
```
# chmod +x evaluate.sh
./evaluate.sh
```

# To run all code files , making annotated data from measurement files, YOLO inference, depth & pixel length estimation, length estimatation and accuracy metrics evaluation

```
# chmod +x run.sh
./run.sh
```

