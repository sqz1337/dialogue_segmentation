# Segmentation Pipelines and Evaluation Metrics

This repository provides three segmentation pipelines along with evaluation metrics for calculating Window Diff and PK scores. The pipelines are as follows:

1. **BigARTM with Topic Tiling**: This pipeline uses BigARTM with topic tiling for segmentation.
2. **Sentence BERT Clustering**: This pipeline utilizes Sentence BERT for clustering-based segmentation.
3. **BERTopic + PCA + Clustering**: This pipeline combines BERTopic, PCA, and clustering for segmentation.

**Pipelines use Wiki727 format. In case you want to use it with AMI dataset, set AMI dataset type in the config, then download the dataset from [here](https://github.com/guokan-shang/ami-and-icsi-corpora) (you may need to use our script split_data.sh to split the files into train/test folders). The same you can do with Wiki50 dataset.**

## Running the Pipelines

Before running any pipeline, make sure you have the dataset in wiki_727 format, for ARTM pipeline it should have 2 sub folders: train and test

### BigARTM with Topic Tiling

To run the BigARTM pipeline, use the following command(for example your dataset is wiki_727):

```bash
bash run_artm.sh ~/wiki_727
```

Please be careful with BigARTM pipeline on shared servers since it uses all available CPU cores. To limit CPU usage and be
nice to your colleagues who also uses the same server use NICE utility, for example:

```bash
nice bash run_artm.sh ~/wiki_727
```

### BERTopic
To run the pipeline with BERTopic, use the following command:
```bash
bash run_btopic.sh ~/wiki_727/test
```
Here, we are using only the test part of the dataset since BERTopic does not have a train stage.

### Sentence BERT Clustering
To run the Sentence BERT pipeline, use the following command:
```bash
bash run_sbert.sh ~/wiki_727/test
```
Similar to BERTopic, Sentence BERT pipeline also does not have a train stage, so we use only the test part of the dataset.

## Evaluation Metrics

The evaluation metrics used for these pipelines are Window Diff and PK scores. The results will be generated after running each pipeline.

Please follow the provided commands to run the pipelines and evaluate their performance.

Please adjust the paths according to your specific dataset.

---
## CLI usage

If you would like to go beyond using only bash scripts, you can leverage the Command Line Interface (CLI) provided by main.py script. The CLI offers three main commands: **artm**, **sbert**, and **btopic**.
Let's discuss each in more detail, starting with **artm**.

### ARTM Command

The **artm** command has 4 subcommands: 
1. **vectorize**
2. **train**
3. **predict**
4. **scores**

Let's talk about each one in more details.
#### vectorize
vectorize command is used to vectorize dataset(assumed that dataset is in wiki_727 format) to ARTM specific format, 
which will be used for training or predict stage, without this operation training and predict stages are not possible.

Example of usage:

```bash
python main.py artm vectorize -i ~/wiki_727/train -o ~/train_vectorized -c train
python main.py artm vectorize -i ~/wiki_727/test -o ~/test_vectorized -c test -d ~/train_vectorized/vocab.train.txt
```
please note that command has four input parameters:

*-i, --input* - path to input dataset

*-o, --output* - path there to save output dataset(directory). This directory will have next files: *docword.collection_name.txt,
mapping.collection_name.txt, vocab.collection_name.txt* and directory batches with binary files inside and artm dictionary with
name *dictionary.dict*. Collection_name is user specified parameter(will be described bellow), files under batches dictionary are
ARTM specific files(calculation of them takes some time). Files under txt format is the corpus converted to UCI format and the mapping is
necessary to restore flattened corpus.

Please note that for train we don't use any dictionary(since we create it from the corpus data) it is necessary only for test part of the dataset.

*-c, --corpus* -  corpus name, used by ARTM library, I recommend to use train and test to don't mess up with the files

*-d, --dictionary* - path to the dictionary in UCI format 

#### train
This command is used to train and dump ARTM model with using artm vectorized dataset, uses only 2 parameters, they are described bellow:

*-i, --input* - path to input dataset in artm vectorized format, for example ~/train_vectorized if you used previous command like in example above

*-o, --output* - path to the dir there to dump model

```bash
python main.py artm train -i ~/train_vectorized -o ~/artm_model
```

#### predict

Command used to make predictions of segments for artm vectorized dataset by already trained model

Command has 4 parameters:

*-m, --model_path* - path to the dir with pretrained dumped model

*-i, --input_path* - path to the ARTM vectorized dataset

*-d, --dictionary* - path to the ARTM bin dictionary from train stage

*-o, --output_path* - output path there to save segmentation predictions made by ARTM model.
Predicts will always have name *predicts.pickle*

```bash
python main.py artm predict -i ~/test_vectorized -m ~/artm_model -d ~/train_vectorized/batches/dictionary.dict -o ~/predicts
```

#### scores

Command used to calculate window diff and pk scores on already segmented data. The command has only 2 parameters

*-p, --predicts_path* - path to the predicts file

*-i, --input_path* - path to the test part of dataset(not vectorized)

```bash
python main.py artm scores -i ~/wiki_727/test -p ~/predicts/predicts.picke
```

As you can see interface for artm is a little bit messy, but it could be simplified after some conversation and collective decision

### sbert Command

Sentence bert has only 2 commands and usage of them is quite simple, so I will not spend a lot of words for describing their usage

#### embed

The command is used to calculate embeddings for dataset on which you want to run segmentation. Since this operation takes a lot of time
I decided to make separate function for this. Usage is quite simple

```bash
python main.py sbert embed -i ~/wiki_727/test -o ~/test_embedded
```

functions has only 2 parameters

*-i, --input_path* - path to the input dataset

*-o, --output_path* - path to the dir there to save embedded dataset

#### scores

This functions uses only one parameter, usage is showed bellow. Used to calculate segmentation scores on embedded dataset

```bash
python main.py sbert scores -i ~/test_embedded
```

*-i, --input_path* path to the embedded dataset


### btopic Command

btopic does not have any subcommands, and used for calculation of segmentation scores by BERTopic model, model uses embedded dataset,
so please use sbert embed to create a proper files for btopic

```bash
python main.py btopic -i ~/test_embedded
```

*-i, --input_path* path to the embedded dataset, could be crated by sentence BERT model(see example above)


___

Please note there is also a lot of parameters in configs/config.yaml file. They can affect the model performance so please read
description of the parameters in the yaml file