TE_new
The TE process contains data for six modes, each with a total of 20 faults (d01-d20) and one normal data (d00) (e.g., mode1_d00.csv is the normal data for mode 1).
DA_based_FD is a program for fault diagnosis using 2D-CNN classification network.
"image_source_TE.py" is the main program that introduces two modal data of TE, mode1 is used to train the model and mode3 is used to test the model, there is no any transfer learning algorithm, just a case of TE data to train the classifier. 
In this case only normal and 9 types of faults from mode1 and mode3 are used for classification. The modes, fault types and number of faults can be changed and the results will be different.
The network structure and parameters can be adjusted to improve the model results.