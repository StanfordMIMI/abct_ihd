# AbCT_IHD
Code for IHD risk asessment from abdominopelvic CTs + EHR data


## Installing requirements
Run the following in command line:
```
conda create -n abct python=3.6.9
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

