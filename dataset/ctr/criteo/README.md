# How to preprocess Criteo Dataset

1. Download the training dataset from Criteo website (here)[https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz]
2. Ex

```shell
# Download dataset from Criteo website
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz

# extract
tar xf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz

# create train val test split
# and format data to binary format
python preprocess.py
```
