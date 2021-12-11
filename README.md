# Bias Detection in News
`TEAM`: 
    
    PizzaPizza
    
`TEAM NUMBER`
    
    29

`Members`: 

    Zishan Kazi - 2019111031
    Shivang Gupta - 2019101117
    
## Files details
- `Report.pdf`: explains the whole procedure.
- `dataset`: contains the data used fot training and testing (other large data files are not included but are uploaded on the GDrive).
- Baseline
    - `cbow.ipynb`: contains the code for cbow, gensim and glove models for bias detection.
    - `Lstm.ipynb`: contains the code for LSTM model for bias detection.
- Baseline+
    - `elmo.ipynb`: contains the code for ELMo model for bias detection.

## For CBOW
- CBOW embedding and word2index:    
https://drive.google.com/file/d/1-4sDR_UKTB32s5Sfb-lNU6WjTDrAt5jO/view?usp=sharing  
https://drive.google.com/file/d/1-C1PPdFLWCaL2P_63LVhl3ectLWG-4pn/view?usp=sharing
- Gensim:   
https://drive.google.com/file/d/1-47a4TpYQjQSRJLV2CZlGakLI4-fGRXx/view?usp=sharing


## For Elmo
1. make checkpoints folder  - https://drive.google.com/drive/folders/1hMr_o2u2kPaPRQJcX4_vuMRvlU7wHKZ2?usp=sharing
2. embeddings - https://drive.google.com/drive/folders/10elbuvERu-5v-N3x0Dlty_UOvpZw42fV?usp=sharing

## Instructions to run
- Just run the jupyter notebook top to bottom, make sure you have big files (which are uploaded on GDrive).

## Dataset details
SemEval 2019 dataset on news bias: https://zenodo.org/record/1489920

The data is split into multiple files. The articles are contained in the files with names starting with "articles-" (which validate against the XML schema article.xsd). The ground-truth information is contained in the files with names starting with "ground-truth-" (which validate against the XML schema ground-truth.xsd).

The data (filename contains "byarticle") is labeled through crowdsourcing on an article basis. The data contains only articles for which a consensus among the crowdsourcing workers existed. It contains a total of 645 articles. Of these, 238 (37%) are hyperpartisan and 407 (63%) are not.

