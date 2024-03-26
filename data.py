import gc,torch
from sklearn.feature_extraction.text import CountVectorizer
import math,os
import numpy as np
import nltk
from nltk.corpus import stopwords
from utils import load_obj_pkl5,load_obj,save_obj,vocab_filtered_data,decompress_pickle
from nltk import word_tokenize

# nltk.download('punkt')
paper = 'emnlp2022'
model='FoTo'
home_dir = '/home/grad16/sakumar/'+paper+'/'+model
data_dir = '/home/grad16/sakumar/'+paper+'/dataset'

def load_data(data,dtype,data_dir,qs,ext,th,skipgram_embeddings,sg_metric):

  data = data.lower()
  dtype = dtype.lower()
  dir ='/content/data_'+data+'/'+dtype
  os.chdir(data_dir+dir)
  data_preprocessed=load_obj_pkl5("data_preprocessed_"+data+"_"+dtype)
  data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data+"_"+dtype)
  
  if skipgram_embeddings: 
    which_embed = "generated_embeddings_all_datasets"
    # which_embed = "generated_embeddings_sg_full_2"
    # which_embed = "generated_embeddings_"+data+"_"+dtype
    os.chdir('/home/grad16/sakumar/emnlp2022/dataset')
    print('got embeddings from - ',os.getcwd())
    embeddings=load_obj_pkl5(which_embed)
    print('using generated embeddings -',which_embed)
    os.chdir(data_dir+dir)
  else: 
    embeddings=load_obj_pkl5("embeddings_"+data+"_"+dtype)
  # queries_data_dict = decompress_pickle("queries_"+data)#+"_"+str(th)
  queries_data_dict = load_obj_pkl5("queries_data_dict_sg")#+"_"+str(th)
  qs = str(qs)
  keywords = queries_data_dict[qs]['query']
  extend_each_by = queries_data_dict[qs]['extend_each_by']
  # whole_query = queries_data_dict[qs]['whole_query']
  # if ext:
    # extended_keywords_list = queries_data_dict[qs]['extended_keywords_list']
    
  if sg_metric == 'cosine':
    extended_keywords_list = queries_data_dict[qs]['extended_keywords_list_sg_cosine']
    print('using sg cosine')
  elif sg_metric == 'euclidean':
    extended_keywords_list = queries_data_dict[qs]['extended_keywords_list_sg_euclidean']
    print('using sg euclidean')
  else:
    # extended_keywords_list = [[k] for k in keywords]
    extended_keywords_list = queries_data_dict[qs]['extended_keywords_list']
  os.chdir(home_dir)
  return data_preprocessed,data_preprocessed_labels,embeddings,data,keywords,extend_each_by,extended_keywords_list,queries_data_dict[qs]

def get_data_label_vocab(data,lables):
  
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=0,dtype=np.uint8)
  train_vec = vectorizer.fit_transform(preprossed_data).toarray()
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  train_label = np.asarray(train_label)
  print("train_input_vec_shape :",train_vec.shape)

  return train_vec,train_label,id_vocab,preprossed_data,vocab