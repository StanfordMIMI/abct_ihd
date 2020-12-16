# L3 selection 

L3 slice selection from Abdominal CT volume was performed using code from:

Fahdi Kanavati, Shah Islam, Eric O. Aboagye, and Andrea Rockall: “Automatic L3 slice detection in 3D CT images using fully-convolutional networks”, 2018; arXiv:1811.09244.

Code is available in [https://github.com/fk128/ct-slice-detection](https://github.com/fk128/ct-slice-detection).

Their trained 1D-UNet model was used for selection in our cohort. Outputs were manually inspected and adjustments were made to ensure correctness. The predictions were correct in 99.5% of cases.
