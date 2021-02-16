# Improving Document-level Sentiment Analysis with User and Product Context

This is the repo for <strong> COLING 2020 </strong> paper titled "<em> Improving Document-level Sentiment Analysis with User and Product Context </em>". 

## 1. Installation
Firstly you need to install all required libraries:

```angular2
pip install -r requirements.txt
```

## 2. Download datasets
Then ownload datasets from the url below:

```angular2
http://ir.hit.edu.cn/ ̃dytang/paper/acl2015/dataset.7z
```

First unzip the downloaded zip file and move all dataset files to "data/document-level-sa/".

## 3. Training

Use the following code to run the training script:

```
python run_document_level_sa.py --task_name yelp-2013 \
    --model_type bert \
    --model_size base \
    --epochs 2 \
    --incremental \
    --do_train \
    --weight_decay 0.0 \
    --learning_rate 3e-5 \
    --warmup_steps 0.1 \
    --max_seq_length 512 \                            
```

## 4. Evaluation

Use the following code to run the evaluation script to evaluate a trained model specified by the given parameters:

```
python run_document_level_sa.py --task_name yelp-2013 \
    --model_type bert \
    --model_size base \
    --epochs 2 \
    --incremental \
    --do_eval \
    --weight_decay 0.0 \
    --learning_rate 3e-5 \
    --warmup_steps 0.1 \
    --max_seq_length 512 \                            
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International Licence](http://creativecommons.org/licenses/by/4.0/).

