# medinfo2019-sa-risk
This repository contains all files related to the MedInfo2019 submission "Text Classification to Inform Suicide Risk Assessment in Electronic Health Records". This includes all code to run the experiments and the annex to the article (annex.pdf).

* **classifier_tune_test.py:** main script to tune and test the classifier.
* **data_preparation.py:** script to query the CRIS database and prepare training and test data.
* **output_results.py:** output all classification tuning and test results for plotting.
* **plot_coeffs.py:** plot the SVM's top features used in classifcation.
* **plot_word_feature_counts.py:** plot word frequency statistics over time.
* **sigtest.py:** run significance testing (McNemar's test) on classification output.
