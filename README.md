# Tmall Repurchase Prediction
Final project for cs182, 24 fall, ShanghaiTech

## Environment
It is recommended using conda to create an isolated Python environment, and install the required dependencies via the 'requirements.txt' file.

Suggest to use python=3.9.x
The code may not work correctly or be very slow on other version.

'''bash
1.Create and activate conda env
conda create -n your_env_name python=3.9
conda activate your_env_name
2.Install the dependencies
pip install -r requirements.txt

**Note:** The environment may not work correctly for GPU on your computer. You could fix it by yourself.

## Directory structure
To help you better understand the organization of the project, here is an overview of the project's directory structure.

TmallRepurchasePrediction/
├── backup  
│   └── model.ipynb.back # Before test for add/remove feature, back up for convenient roll back.  
├── code #The main code part  
│   └── model.ipynb  
├── data  
│   ├── data_format1  
│   │   ├── test_format1.csv  
│   │   ├── train_format1.csv  
│   │   ├── user_info_format1.csv  
│   │   └── user_log_format1.csv  
│   └── data_format2  
│       ├── test_format2.csv  
│       └── train_format2.csv  
├── README.md  
├── references # Maybe you could test and learn from other's code on forum  
│   ├── ref_01.ipynb  
│   └── test.ipynb  
├── requirements.txt  
└── submission  
    └── submission.csv # I have tested that the name of this file will not be check.  

## Data download
You could download the datasets on https://tianchi.aliyun.com/competition/entrance/231576/information. Explanation of them is below the download link.