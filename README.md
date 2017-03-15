## Description
This project implements a method for the Story Cloze Test. [1]

## Requirements
see packages in requirements.txt

#### Data
Download the train, eval and test file from [1]  <br />
The files need to be preprocessed with the coreference script (see ./scripts/coreference)


#### Embeddings
You can use any embedding file in w2v format of any dimension. <br />
Tested with Conceptnet Numberbatch [2]



#### Model
You can download pretrained models with different hyperparameter: LINK  <br />
These models are trained on the eval file. The best (emb CN, size 300, neg 1) scores 0.65 on the test file.


## Run
#### Training
    python train.py -h
Print help for the training arguments.


#### Testing
    python test.py -h
Print help for the testing arguments.



<br />
<br />

[1] http://cs.rochester.edu/nlp/rocstories/ <br />
[2] https://blog.conceptnet.io/2016/05/25/conceptnet-numberbatch-a-new-name-for-the-best-word-embeddings-you-can-download/  <br />


