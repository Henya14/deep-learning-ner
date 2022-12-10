|Milestone|Colab badge|
| ----------- | ----------- |
| Milestone 1 (data preparation)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Henya14/deep-learning-ner/blob/main/data_visualization.ipynb) |
| Milestone 2  (basic training and evaluation)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Henya14/deep-learning-ner/blob/main/basic_training.ipynb) |


# Group info ℹ

## Name: NER 💸
## Group members 👨‍👨‍👦‍👦
| Name      | Neptun |
| ----------- | ----------- |
| Élő Henrik Rudolf      | HR2JO3 |
| Zahorán Marcell        | E5ZY9R |
| ~~Császár Kristóf~~       | ~~DPCIMG~~ (Left the group)  |

# About the project 📚
In this project we make an NLP based hungarian NER model using deep neural networks.

# Project files 📃
The data directory holds all the data for training and validation. The model will use the .csv files, but we kept the original .conllup files just in case. 

- `data_visualization.ipynb` contains the downloading and preparation of the training data.
- `basic_training.ipynb` contains the basic training and basic evaluation code

- `data_augmentation.ipynb` contains the code of the data augmentation

- The `docs` directory contains our documentation in `.pdf` and `.docx` format 

- `evaulation.py` is the script for evaluating a model on the datasets

- `training.py` is the script for the model training.

# Running the project 🏃‍♂️
## Installing the enviroment
You can install the python environment for this project by running the `pip3 --no-cache-dir install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116` command in the project's root directory
## Running the jupyter notebooks
You can simply run the jupyter notebook connected to each milestone by clicking on the corresponding badge on the top of this README file or if it suits you better you can open the `.ipybn` files here on github and click the badge there. 

Of course you are also welcome to clone the repo and run the `.ipynb` files locally with jupyter notebook.

## Training
You can train our model by running the `python training.py` command in the project's root directory.
This script will save the trained model each epoch.  

You can follow the training on the command line and on tensorboard by running `tensorboard --logdir runs` command in the project's root directory. 

## Evaluation
You can evaulate the model by using the `python evaluation.py <model_name>[<evaluation_dataset>]` command where `model_name` is the full name of the PyTorch `.pt` model file in the `models` directory and `evaluation_dataset` is the dataset on which you want to evaulate the model, possible values are: `train` for the training dataset, `devel` for the validation dataset and `test` for the test dataset. The default value for `evaluation_dataset` is `test`.
Example:
You want to evaluate the `first_model.pt` on the `test` dataset: `python evaluation.py first_model.pt test`


# Data 📊
The data is from the [NYTK-NerKor](https://github.com/nytud/NYTK-NerKor) github repo. 

The files are annotated using the [CoNLL-U Plus](https://universaldependencies.org/ext-format.html) format.

Some info from the data's repo:

> The fiction subcorpus contains i) novels from MEK (Hungarian Electronic Library) and Project Gutenberg; and ii) subtitles from OpenSubtitles.

> The legal texts come from EU sources: it is a selection from the EU Constitution, documents from the European Economic and Social Committee, DGT-Acquis and JRC-Acquis.

> The sources of the news subcorpus are: Press Release Database of European Commission, Global Voices and NewsCrawl Corpus.

> Web texts contain a selection from the Hungarian [Webcorpus 2.0](https://hlt.bme.hu/en/resources/webcorpus2).

> Wikipedia texts are from the Hungarian Wikipedia. :)



