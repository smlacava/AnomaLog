# <img src="GUI/favicon.ico" width="25" title="hover text">AnomaLog
AnomaLog is a little tool which simplify the analysis of Machine Learning based Intrusion Detection System.


<br><br>

## Intrusion Detection Systems

Intrusion detection systems (IDSs) define an important and dynamic research area for cybersecurity. 

The role of Intrusion Detection System within security architecture is to improve a security level by identification of all malicious and also suspicious events that could be observed in computer or network system. An IDS is program that monitors computers and network systems for malicious activity or policy violations, typically relying on the analysis of events of sequence of events, so it is an important component of infrastructure component mechanisms which needs to be accurate, adaptive and extensible. 

In order to detect malicious activities, an IDS can follow the misuse-based approach which uses signatures and heuristic rules in order to identify known malicious behavior, or anomaly-based approach to identify behavior that is not that of legitimate user, through statistical analysis, knowledge-based rules or machine learning (through clustering or classification algorithms).

In particular, the pros of the misuse-based approach that that it yields to few false positives, is easy to develop and requires less computational resources, while its cons are that it can detect only known malicious activities (attacks already present in the database) and that updating is too much time consuming.

Instead, the pros of the anomaly-based approach is that it can detect unknown malicious activities, while its cons are that it yields to more false positives than the misuse-based one and the time required in training.

<br><br>

## The AnomaLog class

AnomaLog, a trivial tool which allows to create and train an analyzer through a featurized dataset (for example, KDD99), evaluate its performance, and execute automatic analysis on other datasets.

This tool can:
 - Load a classified or unclassified dataset
 - Preprocess a dataset, deleting the samples containing missing values and extracting all the features
 - Automatically split the dataset into training and test set, eventually returning the indexes related to each training and test sample
 - Create and fit a classification model, currently between Decision Tree, Random Forest, Deep Neural Network, K-Nearest Neighbors and Linear SVM, automatically managing the different function to use in the different cases
 - Evaluate the performance of the classifier, evaluating the accuracy value, the confusion matrix, a report, and in the binary case the ROC curve and the AUC value
 - Save and load a classification model and its settings, such as the dummy parameters and the names of the various classes
 - Set a binary or a multiclass setting, and a possible bias to subtract to the normal class scores, to make a trade-off between missed and false alarms
 - Analyze automatically an unclassified dataset, returning the classified one and in which it is possible to choose if see the only anomalous samples or all the samples, the bias to subtract to the normal class, and how many samples needs to be returned

[Here, you can find an example of analysis on a subset of the KDD99 dataset using AnomaLog.](https://colab.research.google.com/drive/1i_quWDwgKqLP3OYmHemZ66yJpegae4zN?usp=sharing)

<br><br>

## The analyzer

In the repository, it is possible to find the code related to an example of interface which uses previously fitted random forest classification models, both in binary and multiclass settings.

It allows to analyze an unclassified dataset, giving as result a .csv file having the same format, adding a column containing the classification labels.

Furthermore, it is possible to set a bias, the multiclass or the binary setting, and the possibility to return the only anomalous samples in the output file.

<br><br>

> This tool is related to the Cybersecurity Technologies and Risk Management exam project, developed during the MSc in Computer Engineering, Cybersecurity and Artificial Intelligence
