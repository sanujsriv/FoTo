import argparse
import torch
import os
import numpy as np
from data import load_data,get_data_label_vocab

from sklearn.feature_extraction.text import TfidfVectorizer
from metrics import get_docs_idx_in_vis,cal_knn,get_colored_topwords,colored_print_Topics,get_cosine_sum_topics
from metrics import get_bigram_coocurring_word_list,plot_fig,plot_relv_irrelv_docs,cal_AUPR
from utils import get_embedding_tensor,get_topwords,print_Topics,flatten_list,list_of_tensors_to_tensor,cosine_keywords,DESM_score_Corpus
import random
import pandas as pd
from model import FoTo
from train_evaluation import train,test
import gc
import pickle5
import bz2
import pickle
import _pickle as cPickle
from time import time
import torch.nn as nn 
cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
from ranking import get_ranking_parameters

# seed=2022
# random.seed(2022)
# np.random.seed(seed=2022)

def compressed_pickle(data,title):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
 data = bz2.BZ2File(file+".pbz2", 'rb')
 data = cPickle.load(data)
 return data

parser = argparse.ArgumentParser(description='FoTo')

## data arguments
parser.add_argument('-d','--dataset', type=str, default='bbc', help='name of corpus')
parser.add_argument('-dt','--dtype', type=str, default='short', help='for short text')
parser.add_argument('-path','--data_path', type=str, default='./content', help='directory containing data')
parser.add_argument('-bs','--batch_size', type=int, default=250, help='batch size for training')
parser.add_argument('-r','--run', type=int, default=1, help='run')
parser.add_argument('-maxFeat','--max_features', type=int, default=4000, help='max features in countvectorizer (how large should be the vocab)')
parser.add_argument('-qs','--queryset', type=int, default=1, help='the queryset to pass')
# parser.add_argument('-ext','--extended', type=int, default=1, help='to use extended list of keywords')
parser.add_argument('-th','--threshold', type=float, default=0.5, help='to use extended list of keywords')

## model arguments
parser.add_argument('-k','--num_topic', type=int, default=10, help='number of topics')
parser.add_argument('-sg_emb','--skipgram_embeddings', type=int, default=0, help='whether use of skipgram embeddings or any other embeddings')
parser.add_argument('-sg_metric','--skipgram_metric', type=str, default='cosine', help='whether use of skipgram embeddings or any other embeddings')

parser.add_argument('-emb_sz','--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('-act','--activation', type=str, default='relu', help='which activation function(relu,softplus,leaky_relu,sigmoid)')
parser.add_argument('-h1','--hidden1', type=int, default=100, help='dim of hidden layer 1')
parser.add_argument('-h2','--hidden2', type=int, default=100, help='dim of hidden layer 2')
parser.add_argument('-varx', '--variance_x', type=float, default=1.0, help='variance of x {doc coord}')

## optimization arguments
parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('-e','--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('-ncoord','--num_coordinate', type=int, default=2, help='num of coordinates')
parser.add_argument('-drop','--dropout', type=float, default=0.2, help='dropout rate on the encoder')

### evaluation / visualization arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--visualize', type=bool, default=True, help='produce visualization')
parser.add_argument('--show_knn', type=bool, default=False, help='Show KNN score{k = 10,20,30,40,50}')

args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" #"cuda"
print('device: ',device)

if __name__ == '__main__':

  torch.cuda.empty_cache()

  bs = args.batch_size
  epochs = args.epochs
  print('args.epochs',args.epochs)
  activation = args.activation
  # activation = 'softplus'

  en1_units_x = args.hidden1
  en2_units_x = args.hidden2
  dropout = args.dropout
  num_coordinate = args.num_coordinate
  variance_x = args.variance_x
  num_topic = args.num_topic

  skipgram_embeddings = args.skipgram_embeddings
  sg_metric = args.skipgram_metric
  # skipgram_embeddings = 1
  data_name= args.dataset # wos,bbc,searchsnippet,stackoverflow,agnews120k
  
  # if data_name == 'bbc': bs = 250
  # elif data_name == 'searchsnippet': bs = 250
  # elif data_name == 'yahooanswers': bs = 1000
  # elif data_name == 'nfcorpus': bs = 250
  # elif data_name == 'opinions_twitter': bs = 250
  # else: bs = args.batch_size

  # eps_samples = args.eps_samples

  visualize = args.visualize
  show_knn = args.show_knn
  num_words = args.num_words
  emb_size = args.emb_size
  learning_rate = args.learning_rate

  queryset = args.queryset
  # ext = int(args.extended)
  ext=1
  th=0.5

  # home_dir = os.getcwd()
  paper = "emnlp2022"
  model_name = 'FoTo'
  home_dir = '/home/grad16/sakumar/'+paper+'/'+model_name
  data_dir = '/home/grad16/sakumar/'+paper+'/dataset'
  save_dir_no_bkp = '/home/student_no_backup/sakumar/'+paper+'/'+model_name
  # save_dir_no_bkp = home_dir
  #### Data Downloading ####
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print('bs: ',bs)
  dtype=args.dtype 

  # ##### Data loading #####
  # perc_vocab = 0.7
  loaded_data = load_data(data_name,dtype,data_dir,queryset,ext,th,skipgram_embeddings,sg_metric)
  data_preprocessed,data_preprocessed_labels,embeddings,name,keywords,extend_each_by,extended_keywords_list,queries_data_dict = loaded_data
  print(name,len(data_preprocessed_labels),len(data_preprocessed),len(embeddings),dtype,' keyword(s) - ',keywords)

  len_docs = [len(d.split(" ")) for d in data_preprocessed]
  print(np.min(len_docs),np.mean(len_docs).round(2),np.max(len_docs))
  torch.cuda.empty_cache()
  
  train_vec,train_label,id_vocab,preprossed_data_non_zeros,vocab = get_data_label_vocab(data_preprocessed,data_preprocessed_labels)
  
  all_aspects_all_keywords = flatten_list(extended_keywords_list)
  query_as_doc = ' '.join(all_aspects_all_keywords)

  #### DESM
  desm_score = DESM_score_Corpus(all_aspects_all_keywords, train_vec, vocab, embeddings)
  sorted_desm_idx = torch.sort(desm_score,descending=True).indices
  ####

  #### TF-IDF
  tfdifvec = TfidfVectorizer()
  tfdifvec.fit(preprossed_data_non_zeros)
  tfdif_doc_vectors = torch.from_numpy(tfdifvec.transform(preprossed_data_non_zeros).toarray())
  tfdif_query_vectors = torch.from_numpy(tfdifvec.transform([query_as_doc]).toarray())

  tfidf_score = cos_sim(tfdif_query_vectors,tfdif_doc_vectors)
  sorted_tfidf_idx = torch.sort(tfidf_score,descending=True).indices
  
  #### 
  sorted_unique_labels = sorted(set(train_label))

  print('args: '+str(args)+"\n\n")
  print("dropout:",dropout,"\n\n")

  sorted_id_word_vocab = sorted(id_vocab.items(), key=lambda x: x[1]) ### alphabetically sorted
  word_list = [s[1] for s in sorted_id_word_vocab]
  words_tensor = get_embedding_tensor(word_list,embeddings)

  print('keywords: ',keywords)
  embedding_tensor_sorted_alp = get_embedding_tensor(word_list,embeddings)
  embedding_tensor_keywords = get_embedding_tensor(keywords,embeddings)
  all_keywords_score = cosine_keywords(keywords,words_tensor,embeddings)
  
  
  def grouping_relv_scores(a):
    b = [a[:,x]+a[:,x+1] for x in range(a.shape[1]) if x%2==0]
    return list_of_tensors_to_tensor(b).T

  def get_parameters(embedding_tensor_sorted_alp,embedding_tensor_keywords,device):
    emb_size = embedding_tensor_keywords.shape[1]
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
    embedding_tensor_words = embedding_tensor_sorted_alp.to(device)
    embedding_tensor_keywords = embedding_tensor_keywords.to(device)
    return emb_size,cos_sim,embedding_tensor_words,embedding_tensor_keywords  
    
  emb_size,cos_sim,embedding_tensor_words,embedding_tensor_keywords_d = get_parameters(embedding_tensor_sorted_alp,embedding_tensor_keywords,device)
  all_indices = torch.randperm(train_vec.shape[0]).split(bs)
  
  # os.chdir('/home/student_no_backup/sakumar/emnlp2022/FoTo/SavedOutput/searchsnippet/short/topics_50/ext1/sg_eucl/1/qs_1/run_6/')
  # all_indices = decompress_pickle('searchsnippet_short_topics_50_qs_1_run_6_ext_1_all_results')['all_indices']
  num_input = train_vec.shape[1]

  keywords_as_docs,doc_contains_anykey_ext,ranking_q_for_all_doc = get_ranking_parameters(train_vec,preprossed_data_non_zeros,keywords,embeddings,extended_keywords_list,all_keywords_score,word_list,vocab)
  print(doc_contains_anykey_ext)
  
  # if data_name =='yahooanswers': 
  #   keywords_as_docs = keywords_as_docs.reshape(int(len(keywords)/2), 2, len(word_list)).sum(axis=1).astype('int')
  #   print(keywords_as_docs.shape,len(keywords))  
  #   doc_contains_anykey_ext = grouping_relv_scores(doc_contains_anykey_ext)
  #   ranking_q_for_all_doc = grouping_relv_scores(ranking_q_for_all_doc)
  
  num_keyword = len(keywords_as_docs)

  # if data_name == "yahooanswers":
  #   keywords_as_labels = np.asarray(['p{'+keywords[x]+'&'+keywords[x+1]+'}' for x in range(len(keywords)) if x%2==0] )
  # else: 
  
  keywords_as_labels = np.asarray(['p{'+x+'}' for x in keywords])

  print('num_of_latent_keywords: ',num_keyword)
  model = FoTo(num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, dropout, variance_x, bs,
              embedding_tensor_words,embedding_tensor_keywords_d,emb_size,num_keyword,activation, "gaussian") # gaussian , inverse_quadratic

  print("en1,en2,drop,lr,var_x,bs,act - ",en1_units_x,en2_units_x,dropout,learning_rate,variance_x,bs,model.activation,'\n\n')
  
  tstart = time()
  trained_model = train(model,train_vec,train_label,args,all_indices,doc_contains_anykey_ext,keywords_as_docs,ranking_q_for_all_doc,device)
  tstop = time()              
  
  x_list,labels_list,zphi,doc_ids,beta,query_center,ir_query_center = test(trained_model,args,all_indices,train_vec,train_label,doc_contains_anykey_ext,keywords_as_docs,ranking_q_for_all_doc,device)

  bigram_coocurring_word_list = get_bigram_coocurring_word_list(preprossed_data_non_zeros,keywords)
  os.chdir(home_dir)
  save_dir = save_dir_no_bkp+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(args.num_topic)+"/ext1/"+"/qs_"+str(queryset)+"/run_"+str(args.run) # +'/th_'+str(th)+
  os.makedirs(save_dir,exist_ok=True)
  os.chdir(save_dir)

  #*********** Qualitative ***********#
  no_of_topwords = 20
  colored_topics = get_colored_topwords(no_of_topwords,beta,id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list)
  print('Colored topics: <g> extended keyword , <r> keyword, <b> bigram \n\n')
  colored_print_Topics(no_of_topwords,beta,id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list)
  ## focused topics (Qual)

  ## topics quality (Qual)
  print('topics: \n\n')
  print_Topics(beta,id_vocab,no_of_topwords)  
  #**** visualize ****#
    
  figname = data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)+"_ext_"+str(ext)
  # plot_fig(x_list, labels_list,zphi,lim =10,contour='no')
  # lim=20
  number = np.max(x_list)
  lim = round(number/10)*10
  plot_fig(model_name,x_list, labels_list, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
  bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
 
  
  ############
  doc_id_in_seq = np.asanyarray([np.where(doc_ids==i)[0][0] for i in range(len(doc_ids))])
  X_original_seq = x_list[doc_id_in_seq]
  topk = 100

  ### top k Tf-IDf
  topk_tfidfdocs = X_original_seq[sorted_tfidf_idx[:topk]]
  topk_tfidflabels = train_label[sorted_tfidf_idx[:topk]]

  figname = "TFIDF_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)+"_ext_"+str(ext)
  plot_fig(model_name,topk_tfidfdocs, topk_tfidflabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
  bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
  ### /top K Tf-IDf ####


  ### top K DESM ####
  topk_DESMdocs = X_original_seq[sorted_desm_idx[:topk]]
  topk_DESMlabels = train_label[sorted_desm_idx[:topk]]

  figname = "DESM_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)+"_ext_"+str(ext)
  plot_fig(model_name,topk_DESMdocs, topk_DESMlabels, zphi,lim,sorted_unique_labels,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
  bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
  ### /top K DESM ####

  # print('threshold:- ',th)
  #*********** Quantitative ***********#
  # KNN
  knn = cal_knn(x_list,labels_list)
  print('KNN:- ',knn)
  
  # AUPR  
  
  if data_name == 'nfcorpus' or data_name=='opinions_twitter':
    relv_docs = queries_data_dict['ground_truth_docs']
    relv_labels = queries_data_dict['ground_truth_labels']
    d = {item: idx for idx, item in enumerate(data_preprocessed)} 
    relv_docs_idx_ground_truth = [d.get(item) for item in relv_docs]

    sorted_unique_labels_relv = sorted(set(relv_labels))

    figname = "GROUND_TRUTH_"+data_name+"_"+dtype+"_topics_"+str(args.num_topic)+"_run_"+str(args.run)+"_qs_"+str(queryset)+"_ext_"+str(ext)
    plot_fig(model_name,X_original_seq[relv_docs_idx_ground_truth], relv_labels, zphi,lim,sorted_unique_labels_relv,query_center,keywords_as_labels,hv_qwords=True,showtopic=True,
    bold_topics=True,remove_legend=False,show_axis=True,save=True,figname=figname)
    _,_,aupr_ground_truth = cal_AUPR(len(keywords_as_labels),2,relv_docs_idx_ground_truth,X_original_seq,query_center)

    print('AUCPR (ground_truth):- ',aupr_ground_truth)
  
  relv_docs_idx_DESM = sorted_desm_idx[:topk]
  _,_,aupr_DESM = cal_AUPR(num_keyword,num_coordinate,relv_docs_idx_DESM,X_original_seq,query_center)
  print('AUCPR (DESM):- ',aupr_DESM)

  relv_docs_idx_tfidf = sorted_tfidf_idx[:topk]
  _,_,aupr_tfidf = cal_AUPR(num_keyword,num_coordinate,relv_docs_idx_tfidf,X_original_seq,query_center)
  print('AUCPR (tf-idf):- ',aupr_tfidf)


  ## sum avg cosine ##
  ntopwords=10
  all_topics = get_topwords(beta,id_vocab,ntopwords)
  topics_wordlist = [t.split(': ')[1].split(' ') for t in all_topics]
  sum_avg_cos = get_cosine_sum_topics(topics_wordlist,embeddings,keywords)
  
  print('\n\nSum avg cosine for all topics (10 topwords): \n\n',sum_avg_cos)
  with open("results_"+data_name+"_"+str(args.num_topic)+".txt","w") as f:
      f.write(str(args)+"\n\n")
      f.write("Dropout: "+str(dropout)+", topics: "+str(args.num_topic))
      f.write('---'*30+'\n\n')
      f.write('runtime: - '+str(tstop-tstart)+'s\n\n')
      f.write('---------------Printing the Topics------------------\n')
      topword_topics = get_topwords(beta,id_vocab,num_words)
      topword_topics_list=[]
      for topwords in topword_topics:
          topword_topics_list.append(topwords.split())
          f.write(topwords+'\n')
      f.write('---------------End of Topics---------------------\n')
      f.write('KNN:- '+str(cal_knn(x_list,labels_list)))
      f.write('---'*30+'\n\n')


  all_results = {}
  all_results['keywords'] = keywords
  all_results['args'] = args
  all_results['X'] = x_list
  all_results['labels_list'] = labels_list
  all_results['query_center'] = query_center
  all_results['phi'] = zphi
  all_results['beta'] = beta
  all_results['all_indices'] = all_indices
  all_results['doc_ids'] = doc_ids
  all_results['runtime'] = tstop-tstart
  all_results['topics'] =  all_topics
  all_results['colored_topics'] =  colored_topics
  all_results['KNN'] = knn
  all_results['aupr_DESM'] = aupr_DESM
  all_results['aupr_tfidf'] = aupr_tfidf
  if data_name == 'nfcorpus' or data_name=='opinions_twitter':
    all_results['aupr_ground_truth'] = aupr_ground_truth
  all_results['sum_avg_cos'] = sum_avg_cos

  model_signature=data_name+'_'+dtype+'_topics_'+str(args.num_topic)+'_qs_'+str(queryset)+"_run_"+str(args.run)+"_ext_"+str(ext)
  torch.save(trained_model.state_dict(), model_signature+'.pt')
  compressed_pickle(all_results,model_signature+'_all_results')
  os.chdir(home_dir)
