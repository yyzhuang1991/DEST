# DEST

This repository contains the PyTorch codes and a pretrained model for the paper: [Affective Event Classification with Discourse-enhanced Self-training](https://aclanthology.org/2020.emnlp-main.452/)

## 1. Set Up Environment

Set a virtual environment with python3 and install the packages: 
    
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt 


## 2. Predict with a Trained Aff-BERT

In this section, we show how to predict polarities for event phrases using a trained Aff-BERT. 

### 2.0  [Optional] Download the Trained Aff-BERT 

We release a trained Aff-BERT for researchers to predict polarities for event phrases. This Aff-BERT was trained with DEST over all 1,500 labeled events in our TWITTER dataset. You could download the [trained Aff-BERT here](https://drive.google.com/file/d/16TRcDWjmKQ7PaXmR46_4LYIqh6eI-UdU/view?usp=sharing). Please unzip it and put it anywhere you want. 

If you want to train and use a model by yourself, please refer to Section 3 below. 

### 2.1 Make Predictions

The following command predicts the polarity of a single event phrase using a trained Aff-BERT.

      
    cd codes 

    python run.py \
      --predict \
      --predict_str "I break my leg" \
      --checkpoint_dir <PATH/TO/TRAINED_MODEL_DIRECTORY>

Output below shows the prediction using the trained Aff-BERT we release: 
       
``` 
    INPUT: I break my leg
    OUTPUT: neg
    SCORE: pos 0.19, neg 99.57, neu 0.24
```

Arguments: 
   
   * `--predict`: whether to make predictions with a trained Aff-BERT 
   * `--predict_str`: a string to predict for
   * `checkpoint_dir`: the directory that contains the trained Aff-BERT to predict with

When you have a bunch of phrases, it is more convenient to store them in an input file where each line is a phrase. Then you could make predictions over the file using the argument `--predict_infile`. The following command makes predictions for phrases in the sample file `../sample_data/to_predict.txt`:   

    
    python run.py \
    --predict \
    --predict_infile ../sample_data/to_predict.txt \
    --checkpoint_dir <PATH/TO/TRAINED_MODEL_DIRECTORY>

And the prediction would be stored in an output file. 


## 3. How to Train the Model Yourself


### 3.1 Download Data Files
First, please download and unzip the data folder at [the google drive here](https://drive.google.com/drive/folders/1TjOL-99uDzYuSM-QVNFJ_OhakQ_gQpKZ?usp=sharing). These data files are necessary for running the discourse-enhanced Self-training algorithm (DEST). Then put the folder `data/` under the directory `codes`. Details of the data files are provided below: 

   * `processed_labeled_data/`: This contains the preprocessed TWITTER dataset consisting of events with manually annotated polarity labels. There are two subdirectories: 
        
       * ```10folds/```: This contains the train-dev-test splits of the 10-fold cross validation experiments used in our paper.  
       * ```all_data/```: This contains all 1,500 labeled events, in case you are interested in training a model with all labeled data. 
      
      **Data Format** : Labeled data is stored in files named `labeled_events.json`. These files are dictionaries created using the JSON module. The key in the dictionary is an event phrase and the value is its corresponding polarity label (e.g., `{"he kiss my forehead": "pos", "I like his speech": "pos"}`)     
   
   * `unlabeled_data/`: This contains the unlabeled data we used in DEST. It has the following data files: 
      * `unlabeled_event2sentis.json`: It contains unlabeled events with their coreferent sentiment expressions we collected from TWITTER.  This is a dictionary file where the key is an event phrase and the value is a set of sentiment expressions. 

         Note: If you are interested in reproducing this file, you'll need to reimplement everything mentioned in Section 4 in our paper.
   
      * `senti2polar.json`: This file contains the polarity scores for coreferent sentiment expressions. The data is a list of tuples of (`sentiment expression`, `polarity score vector`), where a polarit score vector is a 3-D vector with probilities for Positive, Negative and Neutral. 
        
         Note: To generate polarity scores for these sentiment expressions, we fine-tuned a BERT-based-uncased model over the training data in [SemEval 2017 Task 4](https://alt.qcri.org/semeval2017/task4/). If you are interested in reproducing this model, please refer to Section 3.2 and 3.3 in our paper for more training details (we trained this model using the same hyperparameters as we did for Aff-BERT).

### 3.2 Train Aff-BERT

The following command runs an experiment to train Aff-BERT with DEST over the training set in `fold0` and the unlabeled dataset, and then test it over the test set in `fold0`. 

    cd codes/ 

    fold=0

    python run.py \
      --train \
      --train_dir data/processed_labeled_data/10folds/fold"$fold"/train \
      --dev_dir data/processed_labeled_data/10folds/fold"$fold"/dev \
      --model_save_dir model_output \
      --iter 10 \
      --lr 1e-5 \
      --epoch 5 \
      --threshold 0.95 \
      --neu_threshold 0.9 \
      --unlabeled_event2sentis data/unlabeled_data/unlabeled_event2sentis.json \
      --senti2polar data/unlabeled_data/senti2polar.json \
      --test \
      --test_dir data/processed_labeled_data/10folds/fold"$fold"/test \
      --checkpoint_dir model_output 
 
Arguments: 
    
   * `--train`: whether to train Aff-BERT
   * `--train_dir`: the directory that contains the preprocessed training data 
   * `--dev_dir`: the directory that contains the preprocessed development data 
   * `--model_save_dir`: the director where the trained model would be saved
   * `--iter`: maximum number of iterations to run DEST.
   * `--threshold`: the threshold to choose the newly labeled data based on the joint score computed in DEST
   * `--neu_threshold`: the threshold to choose extra newly labeled neutral data based on the score by Aff-BERT
   * `--unlabeled_event2sentis`: the released dictionary file that contains unlabeled events with their coreferent sentiment expressions
   * `--senti2polar`: the released file that contains sentiment expressions with their polarity scores
   * `--test`: whether to test Aff-BERT over some test data 
   * `--test_dir`: the directory that contains the preprocessed test data (labeled_events.json)
   * `--checkpoint_dir`: the directory that contains the trained model that is to be tested

As the TWITTER dataset is small (only 1,500 events), it is better to train Aff-BERT with DEST over all labeled data if we want to use the model to predict over unseen data later. The following command trains an Aff-BERT with DEST over all labeled data (this means we have no development data to select the best model during training, and no test data to evaluate over afterwards) : 
    
    cd codes/

    python run.py \
        --train \
        --train_dir data/processed_labeled_data/all_data \
        --model_save_dir model_output \
        --iter 10 \
        --lr 1e-5 \
        --epoch 5 \
        --threshold 0.95 \
        --neu_threshold 0.9 \
        --unlabeled_event2sentis data/unlabeled_data/unlabeled_event2sentis.json \
        --senti2polar data/unlabeled_data/senti2polar.json \


## Citation

Please cite our work if you use it in your research. 
    
    @inproceedings{zhuang-etal-2020-affective,
    title = "Affective Event Classification with Discourse-enhanced Self-training",
    author = "Zhuang, Yuan  and
      Jiang, Tianyu  and
      Riloff, Ellen",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.452",
    doi = "10.18653/v1/2020.emnlp-main.452",
    pages = "5608--5617",
}

 
   