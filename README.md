# Bias Detection in News

`TEAM`: 
    
    PizzaPizza  

`Members`: 

    Zishan Kazi
    Shivang Gupta

## Files details
- `docs`: contains the project write up, report.pdf, etc.
- `dataset`: contains the data used fot training and testing (other large data files are not included but are uploaded on the GDrive).
- `cbow.ipynb`: contains the code for cbow, gensim and glove models for bias detection.

## Instructions to run
- Just run the jupyter notebook top to bottom, make sure you have big files (which are uploaded on GDrive).

## Dataset details
SemEval 2019 dataset on news bias: https://zenodo.org/record/1489920

The data is split into multiple files. The articles are contained in the files with names starting with "articles-" (which validate against the XML schema article.xsd). The ground-truth information is contained in the files with names starting with "ground-truth-" (which validate against the XML schema ground-truth.xsd).

The data (filename contains "byarticle") is labeled through crowdsourcing on an article basis. The data contains only articles for which a consensus among the crowdsourcing workers existed. It contains a total of 645 articles. Of these, 238 (37%) are hyperpartisan and 407 (63%) are not.
