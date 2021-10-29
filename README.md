# NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data
This repository contains the code and the data to train [NetTCR-2.0](https://www.nature.com/articles/s42003-021-02610-3) model. 
## License 
NetTCR-2.0 is developed by Morten Nielsen's group at the Technical University of Denmark (DTU).
NetTCR-2.0 code and data can be used freely by academic groups for non-commercial purposes.
If you plan to use NetTCR or any data provided with the script in any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk). 

For scientific questions, please contact Morten Nielsen (mniel@dtu.dk).
## Data
This data folder contains the data files used to train NetTCR-2.0.

File description:
- **train_beta_{90,92,94,99}**: CDR3b only *training* dataset, partitioned using {90,92,94,99}% partitioning threshold;

- **mira_eval_threshold{90,92,94,99}.csv**: MIRA dataset, used for the *evaluation* of the CDR3b models. The threshold refers to the separation from the training set; 

- **train_alphabeta_{90,95}.csv**: Paired alpha beta dataset. The partitioning is done using {90,95}% partitioning threshold and using the average similarity between alpha and beta chain;

- **train_ab_{90,95}_{alpha,beta,aphabeta}.csv**: Paired alpha beta dataset. The partitioning is based on the {alpha,beta, alphabeta} chain(s) and is done using {90,95}% partitioning threshold;

- **ext_eval_paired_data.csv**: Paired TCRs sequences of *external evaluation* of the alpha+beta model.

## Train networks

You can train the NetTCR_ab models running

`python nettcr.py --trainfile test/sample_train.csv --testfile test/sample_test.csv`

This will print the predictions on the standard output or on a file (that can be specified with the option --outfile).

Both training and test set should be a comma-separated CSV files. The files should have the following columns (with headers): CDR3a, CDR3b, peptide, binder (the binder coulmn is not required in the test file). 
See test/sample_train.csv and test/sample_test.csv as an example.

## NetTCR server
NetTCR-2.0 is also availavble as a web server at https://services.healthtech.dtu.dk/service.php?NetTCR-2.0.
The server offers the possibility to evaluate pre-trained models on new data. See Instructions tab for more information.

## Citation
Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021). https://doi.org/10.1038/s42003-021-02610-3

