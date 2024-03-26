from collections import defaultdict
import torch.nn as nn 
import numpy as np
import torch

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
torch.pi = torch.acos(torch.zeros(1)).item() * 2 

def flatten_list(user_list): return [item for sublist in user_list for item in sublist]
def get_embedding_tensor(word_list,embeddings): return torch.tensor([embeddings[w] for w in word_list]) 
def cosine_sqrt(keyword_torch,words_tensor): return 1 - ((1 - cos_sim(keyword_torch,words_tensor) )/2)**0.5


def toT(a): return torch.tensor(a)
def DESM_score(query_list, doc, word_list, embeddings):
  cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
  num_voc = doc.shape[0]
  doc_bar = torch.zeros(300)
  doc_length = 0
  for v in range(num_voc):
    if(doc[v]):
      doc_bar.add_(doc[v] * torch.from_numpy(embeddings[word_list[v]])/torch.norm(torch.from_numpy(embeddings[word_list[v]]))) 
      doc_length = doc_length + doc[v]
  doc_bar = doc_bar / doc_length
  sum = 0

  for q in query_list:
    keyword = query_list[0]

    cosine_similarity_q_keyword = cos_sim(toT(embeddings[q]),toT(embeddings[keyword]))
    q_doc_bar_dot_product = torch.dot(torch.from_numpy(embeddings[q]) , doc_bar)
    norm_q_doc_bar = torch.norm(torch.from_numpy(embeddings[q]))*torch.norm(doc_bar)

    sum += (cosine_similarity_q_keyword * q_doc_bar_dot_product) / norm_q_doc_bar
  # # sum = sum/len(query_list) 

  #   sum = (q_doc_bar_dot_product) / norm_q_doc_bar
  # sum = sum/len(query_list)
  return sum


def get_ranking_parameters(train_vec,preprossed_data_non_zeros,keywords,embeddings,extended_keywords_list,all_keywords_score,word_list,vocab):
  
  keywords_as_docs = np.zeros(shape=(len(keywords),len(word_list)),dtype=np.uint8)

  extended_keywords_list_idx = []
  if len(extended_keywords_list)==len(keywords):
    for extended_keyword in extended_keywords_list:
        extended_keywords_list_idx.append([vocab[w] for w in extended_keyword])
  else: 
    extended_keywords_list_idx.append([vocab[w] for w in extended_keywords_list])
    
  doc_contains_anykey_ext = torch.zeros(train_vec.shape[0],len(keywords))
  # DESM_doc_contains_anykey_ext = torch.zeros(train_vec.shape[0],len(keywords))

  for i in range(len(keywords)):
    kws = extended_keywords_list[i]
    print(kws)
    for w in kws:
      keywords_as_docs[i][word_list.index(w)] += 1 
    for d in range(train_vec.shape[0]):
      # if(train_vec[d][extended_keywords_list_idx[i]].sum()!=0):
      #   DESM_doc_contains_anykey_ext[d][i] = DESM_score(kws,train_vec[d],word_list,embeddings)
      # doc_contains_anykey_ext[d][i] += train_vec[d][vocab[w]]
      
      # cosine
      doc_contains_anykey_ext[d][i] = torch.tensor(train_vec[d][extended_keywords_list_idx[i]] * all_keywords_score[i][extended_keywords_list_idx[i]].numpy()).sum()

  ## count of extended keywords metric (kxN)
  count_for_d = []
  for d in range(train_vec.shape[0]):
    count = [0]*len(extended_keywords_list_idx)
    for i in range(len(extended_keywords_list_idx)):
      
      # word_count_ext_k = train_vec[d][extended_keywords_list_idx[i]].sum()
      # word_count_ext_k = word_count_ext_k.astype('float32')
      
      word_count_ext_k = (train_vec[d][extended_keywords_list_idx[i]] * all_keywords_score[i][extended_keywords_list_idx[i]].numpy()).sum()
      count[i] = word_count_ext_k
    count_for_d.append(count)
  score_q_for_all_doc = torch.tensor(count_for_d)
  
  bpr1 =  doc_contains_anykey_ext # DESM_doc_contains_anykey_ext
  bpr2 = score_q_for_all_doc
  
  return keywords_as_docs,bpr1,bpr2