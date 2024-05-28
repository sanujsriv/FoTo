# FoTo - A visual focused topic model for targeted analysis of short texts

FoTo is a novel targeted visual topic model which can extract and visualize topics and documents relevant to targeted aspects for focused analysis.

Link to the paper:- https://aclanthology.org/2024.lrec-main.653/

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
Here is an example visualization produced by FoTo for searchsnippet - <br/>

![ssnip_vis](/visualization/main_visualization_FoTo_example_searchsnippet.png)

# Citation 
```
@inproceedings{kumar-le-2024-foto-targeted,
    title = "{F}o{T}o: Targeted Visual Topic Modeling for Focused Analysis of Short Texts",
    author = "Kumar, Sanuj  and
      Le, Tuan",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.653",
    pages = "7406--7416",
    abstract = "Given a corpus of documents, focused analysis aims to find topics relevant to aspects that a user is interested in. The aspects are often expressed by a set of keywords provided by the user. Short texts such as microblogs and tweets pose several challenges to this task because the sparsity of word co-occurrences may hinder the extraction of meaningful and relevant topics. Moreover, most of the existing topic models perform a full corpus analysis that treats all topics equally, which may make the learned topics not be on target. In this paper, we propose a novel targeted topic model for semantic short-text embedding which aims to learn all topics and low-dimensional visual representations of documents, while preserving relevant topics for focused analysis of short texts. To preserve the relevant topics in the visualization space, we propose jointly modeling topics and the pairwise document ranking based on document-keyword distances in the visualization space. The extensive experiments on several real-world datasets demonstrate the effectiveness of our proposed model in terms of targeted topic modeling and visualization.",
}

```
