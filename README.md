# NetTCR-2.0
NetTCR-2.0. Sequence-based prediction of peptide-TCR binding

This repo contains the data files used to train NetTCR-2.0.

File description:
- **train_beta_{90,92,94,99}**: CDR3b only *training* dataset, partitioned using {90,92,94,99}% partitioning threshold;

- **mira_eval_threshold{90,92,94,99}.csv**: MIRA dataset, used for the *evaluation* of the CDR3b models. The threshold refers to the separation from the training set; 

- **train_alphabeta_{90,95}.csv**: Paired alpha beta dataset. The partitioning is done using {90,95}% partitioning threshold and using the average similarity between alpha and beta chain;

- **train_ab_{90,95}_{alpha,beta,aphabeta}.csv**: Paired alpha beta dataset. The partitioning is based on the {alpha,beta, alphabeta} chain(s) and is done using {90,95}% partitioning threshold;

- **ext_eval_paired_data.csv**: Paired TCRs sequences of *external evaluation* of the alpha+beta model.

