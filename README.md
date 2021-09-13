# Improving Document-level Sentiment Analysis with User and Product Context

This is the repo for <strong> COLING 2020 </strong> paper titled "<em> Improving Document-level Sentiment Analysis with User and Product Context </em>". 

## 1. Installation
Firstly you need to install all required libraries:

```angular2
pip install -r requirements.txt
```

## 2. Download datasets
Then download datasets from this url: [dataset](https://drive.google.com/file/d/1Bdt_jw-kiZCt7vJyfXe1hYmPKMinbtFu/view?usp=sharing).


Then unzip the downloaded zip file and move all dataset files to "data/document-level-sa-dataset/".

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

## Citation

```bibtex
@inproceedings{lyu-etal-2020-improving,
    title = "Improving Document-Level Sentiment Analysis with User and Product Context",
    author = "Lyu, Chenyang and Foster, Jennifer and Graham, Yvette",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.590",
    doi = "10.18653/v1/2020.coling-main.590",
    pages = "6724--6729"
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International Licence](http://creativecommons.org/licenses/by/4.0/).

