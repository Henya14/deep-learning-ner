[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Henya14/deep-learning-ner/blob/main/data_visualization.ipynb)
# Group info
Name: NER
## Group members
| Name      | Neptun |
| ----------- | ----------- |
| Élő Henrik Rudolf      | HR2JO3 |
| Zahorán Marcell        | E5ZY9R |
| Császár Kristóf        | DPCIMG |

# About the project
In this project we make an NLP based hungarian NER model using deep neural networks.

# Project files
The data directory holds all the data for training and validation. The model will use the .csv files, but we kept the original .conllup files just in case. 

data_visualization.ipynb contains the downloading and preparation of the training data.


# Data 
The data is from the [NYTK-NerKor](https://github.com/nytud/NYTK-NerKor) github repo. 

The files are annotated using the [CoNLL-U Plus](https://universaldependencies.org/ext-format.html) format.

Some info from the data's repo:

> The fiction subcorpus contains i) novels from MEK (Hungarian Electronic Library) and Project Gutenberg; and ii) subtitles from OpenSubtitles.

> The legal texts come from EU sources: it is a selection from the EU Constitution, documents from the European Economic and Social Committee, DGT-Acquis and JRC-Acquis.

> The sources of the news subcorpus are: Press Release Database of European Commission, Global Voices and NewsCrawl Corpus.

> Web texts contain a selection from the Hungarian [Webcorpus 2.0](https://hlt.bme.hu/en/resources/webcorpus2).

> Wikipedia texts are from the Hungarian Wikipedia. :)