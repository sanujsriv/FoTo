# FoTo - A visual focused topic model for targeted analysis of short texts

This is the code for FoTo.

FoTo is a novel targeted visual topic model which can extract and visualize topics and documents relevant to targeted aspects for focused analysis.

# Environment
Tested on:-


# Preprocessing
The scripts used for preprocessing the data can be found in the folder "preprocessing".

## Datasets
The preprocessed example dataset can be found in the "content" folder. It follows the directory structure - content/data_{dataname}/short/**.pkl**. 

## Data Preprocessing
If you want to run WTM on your own dataset you can follow the script for preprocessing i.e. **preprocessing.py** or use your own. It is recommended to follow the preprocessing steps given in the **preprocessing.py**

## Generating Embeddings
WTM can also run on generated embeddings. It uses *skipgram* technique by default to generate embeddings.The script for generating the embeddings could be found in the **preprocessing.py**. You can either follow that or generate your own embeddings.

# Running FoTo
You can directly pass *bbc* to run FoTo on bbc dataset.
```  
python3 main.py --dataset bbc --num_topics 10 -e 1000 -drop 0.2
```

## Running FoTo
For running FoTo use the following script -

```
python3 run_script_FoTo.py --data_name bbc --queryset 1 --num_topic 10
```
<This script will run FoTo on bbc dataset with 10 topics for the first query> <br/>

# Visualizations
Here is an example visualization produced by FoTo for  - <br/>

![ssnip_vis](/visualizations/searchsnippet_WTM.png)
![ssnip_label](/visualizations/searchsnippet_WTM_label.png)

# Citation 
```
```
