
## Dataset Splitting
The in-silico RNA secondary structure datasets are obtained from the existing works, 
where the reference has been properly cited in the main text (Section 1.1). 
As a results, these dataset splits has been defined in their original publications and acknowledged by the community. 
For the RNA translation efficiency prediction and region classification tasks, we split the dataset into training, validation, and test sets,
with a ratio of 80:10:10(%). 


## Mitigation to Data Leakage
Data leakage is a common issue in machine learning, leading to inflated performance metrics. 
To avoid data leakage, we evaluate the downstream tasks with the sequences that are not included in the OneKP database,
such as the arabidopsis and rice sequences. This means there is no data leakage in our experiments, 
and it ensures that the model generalizes well to unseen data without overfitting to the training data. 
The datasets of secondary structure prediction tasks are retrieved from the existing works, 
where there is no data leakage issue reported in the literature.


## Performance Metrics
The data imbalances has been a common issue in the field of bioinformatics, which may lead to biased model performance.
As a result, we have used the macro-F1 score performance metrics to evaluate the model performance, 
including the RNA translation efficiency prediction and region classification tasks and the RNA secondary structure prediction task.
The macro-F1 score calculation used in the article is implemented by the sklearn library in Python, 
which is available at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html.

## Pseudo-code
Here we describe the pseudo-code for the PlantRNA-FM model pretraining training paradigm.
```
Step 1: Collect the OneKP database for pretraining.
Step 2: Data preprocessing. We extract the sequences from the OneKP database and preprocess the sequecnes, which requires mutiple sub-steps.
    Step 2.1: Truncate the RNA sequences exceeding 512 nucleotides.
    Step 2.2: Filter out sequences shorter than 20 nucleotides.
    Step 2.3: Annotate the local RNA structures of all RNA sequences using ViennaRNA.
    Step 2.4: Retrieve the CDS, 5' UTR, and 3' UTR sequences.
Step 3: Pretraining objectives formulation and sequence processing. 
We process the sequeces inputs according to each pretraining objective and annoate the outputs for loss calculation.
We conceputalize three pretraining objectives as follows:
    Step 3.1: Masked RNA sequence prediction. The sequecnes has been paritially masked as inputs and the outputs are the masked nucleotides.
    Step 3.2: RNA secondary structure prediction. The sequecnes has been annotated with the RNA secondary structure and the outputs are the annotated structures, i.e., "(, ., )".
    Step 3.3: RNA region classification. The sequecnes has been annotated with the RNA regions (i.e., 5' UTR, CDS and 3'UTR) and the outputs are the annotated regions.
Step 4: Model pretraining. We train the PlantRNA-FM model on the preprocessed data with the formulated pretraining objectives.
Step 5: Model evaluation. We evaluate the model performance on the downstream tasks, including RNA translation efficiency prediction and region classification tasks.
Step 6: Interpretation. We perform considerable experiments for interpreting the model predictions and understanding the functional RNA motifs in plants.
For example, we extract the attention contrast matrix to visualize the important RNA motifs for the RNA translation efficiency in Section 1.3.
```

