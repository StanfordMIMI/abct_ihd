# AbCT_IHD
Code for IHD risk asessment from abdominopelvic CTs + EHR data, from the following paper (under review):

**Opportunistic Assessment of Ischemic Heart Disease Risk Using Abdominopelvic Computed Tomography and Medical Record Data: A Multimodal Explainable Artificial Intelligence Approach**

Authors: Juan M Zambrano Chaves, Andrew L Wentland, Arjun D Desai, Imon Banerjee, Gurkiran Kaur, Ramon Correa, Robert D Boutin, David J Maron, Fatima Rodriguez, Alexander T Sandhu, R Brooke Jeffrey, Daniel Rubin, Akshay S Chaudhari, Bhavik Patel.



## Installing requirements
Requires Linux. Create anaconda environment by running the following in command line:
```
conda create -n abct python=3.6.9
conda activate abct
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

## Running Inference
To perform IHD risk assessment using an L3 dcm slice, clinical covariates or both:

1. Download the [models from Google Drive](https://drive.google.com/drive/folders/1Gqg3mBEohoWsiVahzgvDLLDd3krIqoO9?usp=sharing) locally.

2. Run the following script with appropriate flags, an example for fusion prediction is shown below:

```
python risk_assessment.py \
-risk_assessment both \
-trained_model_dir ./models/ \
-image_data_dir ./data/l3_slices \
-clinical_data_path ./data/clinical_data.csv \
-output_dir ./predictions/
```

Notes:

The dicom file name should be the anon_id specified in the clinical_data.csv file, i.e. `abcde.dcm` corresponds to `anon_id : abcde`. 

If doing fusion inference, there should be a 1:1 correspondence between .dcm files and data rows in clinical_data.csv file.



