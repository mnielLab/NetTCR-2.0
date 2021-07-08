# NetTCR-2.0
This repository contains the code and the data to train NetTCR-2.0 model.

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

This will print the predictions on screen or on a file (that can be specified with the option --outfile).

## NetTCR server
NetTCR-2.0 is also availavble as a web server at https://services.healthtech.dtu.dk/service.php?NetTCR-2.0.
The server offers the possibility to evaluate pre-trained models on new data. See Instructions tab for more information.

## License 
NetTCR-2.0 is developed by Morten Nielsen's group at the Technical University of Denmark (DTU).

If you plan to use NetTCR or any data provided with the script in any for-profit application, you are required to obtain a separate license. 

For scientific questions, please contact Morten Nielsen (mniel@dtu.dk).
