# AbCT_IHD
Code for IHD risk asessment from abdominopelvic CTs + EHR data, from the following paper (under review):

**Opportunistic Assessment of Ischemic Heart Disease Risk Using Abdominopelvic Computed Tomography and Medical Record Data: A Multimodal Explainable Artificial Intelligence Approach**

Authors: Juan M Zambrano Chaves, Andrew L Wentland, Arjun D Desai, Imon Banerjee, Gurkiran Kaur, Ramon Correa, Robert D Boutin, David J Maron, Fatima Rodriguez, Alexander T Sandhu, R Brooke Jeffrey, Daniel Rubin, Akshay S Chaudhari, Bhavik Patel.



## Installing requirements
Requires Linux or MacOS. Create anaconda environment by running the following in command line:
```
conda create -n abct python=3.6.9
conda activate abct
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

