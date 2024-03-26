from sklearn.metrics import average_precision_score as AP_score
from sklearn.metrics import precision_recall_curve,auc
import itertools
from sklearn.neighbors import KNeighborsClassifier
from utils import flatten_list,get_topwords,get_embedding_tensor,generate_co_occurrence_matrix,cosine_keywords
import matplotlib.pyplot as plt
import plotly.graph_objects as gos
import seaborn as sb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 

cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

def inverse_cosine_sqrt(cosine_sqrt_val):
  cosine_sim_val = 1 - 2 * ((1 - cosine_sqrt_val)**2)
  return cosine_sim_val

def list_of_tensors_to_tensor(loT):
  stacked_tensor = torch.stack(loT)
  return stacked_tensor


def toT(a): return torch.tensor(a)
### Focused Topics

## Qualitative (more green , blue , red in same line more focused THAT topic is)

def get_colored_topwords(topwords,beta, id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list):
  topic_indx = 0
  all_topics = []
  topic_topwords = 20
  for i in range(len(beta)):
      all_topics.append( str(topic_indx)+": "+ " ".join([id_vocab[j] for j in beta[i].argsort()[:-topic_topwords - 1:-1]]))
      topic_indx+=1

  flattened_ext_keylist = flatten_list(extended_keywords_list)
  for t in range(len(all_topics)):
      topic = all_topics[t].split()[0]
      words_in_topwords = all_topics[t].split()[1:]
      for i in range(len(words_in_topwords)):
        if words_in_topwords[i] in keywords: 
          words_in_topwords[i] = "<r> "+words_in_topwords[i]+" </r>"
        elif words_in_topwords[i] in bigram_coocurring_word_list:
          words_in_topwords[i] = "<b> "+words_in_topwords[i]+" </b>"
        elif words_in_topwords[i] in flattened_ext_keylist:
          words_in_topwords[i]= "<g> "+words_in_topwords[i]+" </g>"
      all_topics[t] = ' '.join([topic]+words_in_topwords)
  return all_topics

def colored_print_Topics(topwords,beta,id_vocab,keywords,bigram_coocurring_word_list,extended_keywords_list):
  flattened_ext_keylist = flatten_list(extended_keywords_list)
  print("---"*10)
  topword_topics = get_topwords(beta, id_vocab,topwords)
  topword_topics_list=[]
  for topwords in topword_topics:
    topic = topwords.split()[0]
    words_in_topwords = topwords.split()[1:]
    print(topic,end=" ")
    for word in words_in_topwords:
      if word in keywords:
        print(' <r>',word,'</r> ',end=" ")
        # print(colored(word,'red'),end=" ") # keyword == red
      elif word in bigram_coocurring_word_list:
        print(' <b>',word,'</b> ',end=" ")
        # print(colored(word,'blue'),end=" ") # print(colored(0, 0, 255, word),end=" ") co-occuring == blue 
        # print(colored(255, 255, 255, word),end=" ") # co-occuring == blue
      elif word in flattened_ext_keylist:
        print(' <g>',word,'</g> ',end=" ")
        # print(colored(word,'green'),end=" ")
        # print(colored(0, 255, 0, word),end=" ") # similar == green
      else: print(word,end=" ") # other words == white
    print('')  
  print("---"*10)

### Quantitative => sum of average of cosine
import plotly.express as px
import plotly.graph_objects as go
def plot_relv_topwords(keywords,np_topics_wordsScores):
  for i in range(len(keywords)): 
    fig = px.histogram(pd.DataFrame(np_topics_wordsScores[i],columns=['scores']), x="scores")
    print(keywords[i])
    fig.show()

def plot_lines(x,y,title,yaxis_title,xaxis_title):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y,
                      mode='lines+markers',
                      name="FoTo",
                line=dict(color='red')))
  fig.update_layout(title=title,
                    yaxis_title=yaxis_title,
                    xaxis_title=xaxis_title)

  fig.show(renderer='colab')

def plot_relative_diff(x,y,y2,y3,name2,name3,title,yaxis_title,xaxis_title):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x, y=y,
                      mode='lines+markers',
                      name="FoTo",
                line=dict(color='red')))
  fig.add_trace(go.Scatter(x=x, y=y2,
                      mode='lines+markers',
                      name=name2,
   line=dict(color='blue')))
  fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines+markers',
                    name=name3,
                     line=dict(color='green')))
  fig.update_layout(title=title,
                    yaxis_title=yaxis_title,
                    xaxis_title=xaxis_title)

  fig.show(renderer='colab')

def keyw_sum_mean_median(np_topics_wordsScores):
   return keywords,torch.max(np_topics_wordsScores,0).values.sum().item() , torch.mean(np_topics_wordsScores,-1) , torch.median(np_topics_wordsScores,-1).values
# print("Spike Slab Filter (top # relv words that represents documents):",spikeSlab_filter,"\n\n")

def get_bigram_coocurring_word_list(data_preprocessed,keywords):
  nlargestVal =  20
  text_data = [d.split(' ') for d in data_preprocessed]
  data = list(itertools.chain.from_iterable(text_data))
  matrix, vocab_index = generate_co_occurrence_matrix(data)
  
  data_matrix = pd.DataFrame(matrix, index=vocab_index,
                              columns=vocab_index)

  bigram_coocurring_word_list = []

  for k in keywords: 
    bigram_coocurring_word = np.array(data_matrix.nlargest(nlargestVal, k, keep='first').index)
    bigram_coocurring_word_list.extend(bigram_coocurring_word)
  return bigram_coocurring_word_list
  
def get_cosine_sum_topics(topics_wordlist,embeddings,keywords):
  np_topics_wordsScores_all = []
  for topics in topics_wordlist:
    topics_wordtensors = get_embedding_tensor(topics,embeddings)
    topics_wordsScores = cosine_keywords(keywords,topics_wordtensors,embeddings)
    np_topics_wordsScores = list_of_tensors_to_tensor(topics_wordsScores)
    np_topics_wordsScores_all.append(np_topics_wordsScores)
  
  sum_avg_cos = []
  for np_topics_wordsScores in np_topics_wordsScores_all:
    sum_avg_cos.append((np_topics_wordsScores.mean(-1)).sum(-1).item())
  sum_avg_cos = np.array(sum_avg_cos)
  return sum_avg_cos

######### Focused Visualization ######## 
## Qualitative (how close are the relv docs to the ground truth)

def plot_relv_irrelv_docs(filename,zx1,zx2,l1,l2, zphi, query_center,query_words,hv_qwords,keywords,lim,contour='No'):
       
    fig, ax = plt.subplots( figsize=(20, 20))
    # if contour=='yes':
    #    get_Contour(ax,zx,lim)
    label_colors_dict = {'direct': 'C1','indirect':'C2',
                         'relevant(T)': 'C1','irrelevant(F)': 'C2'}
    sb.scatterplot(ax=ax,x=zx1[:,0],y=zx1[:,1],hue=l1,palette=label_colors_dict,alpha=0.8,s=50)
    sb.scatterplot(ax=ax,x=zx2[:,0],y=zx2[:,1],hue=l2,palette=label_colors_dict,alpha=0.8,s=50)
    
    ax.set(ylim=(-lim,lim))
    ax.set(xlim=(-lim,lim))

    ax.text(query_center[0],query_center[1], 'X' ,c='black',weight='bold',fontsize=20)
    # ax.text(0,0, 'X' ,c='black')
    
    if hv_qwords:
      for i in range(len(query_words)):
        if (i==len(query_words)-1):
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)
        else:
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)
          
    ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)

    for indx, topic in enumerate(zphi):
        ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13)
    plt.savefig('FoTo_vis_'+filename+".png", bbox_inches='tight')


def get_docs_idx_in_vis(relv_docs,preprossed_data_non_zeros,doc_ids):
  d = {item: idx for idx, item in enumerate(preprossed_data_non_zeros)} 
  doc_ids_list = list(doc_ids)
  relv_docs_idx = [d.get(item) for item in relv_docs]
  docs_in_vis_idx = [doc_ids_list.index(r_i) for r_i in relv_docs_idx]
  assert (doc_ids[docs_in_vis_idx] == relv_docs_idx).all() == True
  return docs_in_vis_idx

## Quantitative 
def cal_AUPR(num_keyword,num_coordinate,relv_docs_in_vis_idx,x_list,query_center):
  num_docs = len(x_list)
  map_true_ranking = torch.zeros(num_docs) 
  map_true_ranking[relv_docs_in_vis_idx] = 1.0

  doc_query_size = (num_docs, num_keyword, num_coordinate)

  x_q = toT(x_list).view(num_docs,1,num_coordinate).expand(doc_query_size)
  q_x = toT(query_center).view(1, num_keyword,num_coordinate).expand(doc_query_size)
  dist_x_q = (x_q - q_x).pow(2).sum(-1)
  minDist_x_q = torch.min(dist_x_q,-1).values
  # minDist_x_q = dist_x_q.sum(-1) ## SUM DISTANCE
   
  precision, recall, _ = precision_recall_curve(map_true_ranking,-minDist_x_q)

  map_true_ranking = map_true_ranking.numpy()
  minDist_x_q = minDist_x_q.numpy()
  return map_true_ranking,minDist_x_q,auc(recall,precision)

## (average precision score for documents close to any of the keywords)

######## Visualization Quality ###############

## Quantitative (clustering quality of visualization)
def cal_knn(coordinate, label):
    output = []
    for n_neighbors in [10, 20, 30, 40, 50]:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        neigh.fit(coordinate, label)
        output.append(neigh.score(coordinate, label))
    return output


## Qualitative (clusters in visualization)

def get_labels_dict(unique_labels):
  labels_dict = {}
  for l in unique_labels:
    labels_dict[l] = 'C'+str(unique_labels.index(l))
  return labels_dict

def plot_fig(model_name,zx, labels_list, zphi,lim,sorted_unique_labels,query_words,keywords,hv_qwords=True,showtopic=False
            ,bold_topics=True,remove_legend=False,show_axis=True,save=False,figname="plot"):
       
    fig, ax = plt.subplots( figsize=(20, 20))
    # if contour=='yes':
    #    get_Contour(ax,zx,lim)

    label_colors_dict = get_labels_dict(sorted_unique_labels+['keywords'])
    # sb.scatterplot(ax=ax,x=zx[:,0],y=zx[:,1],hue=labels_list,alpha=0.8,palette='deep')
    g = sb.scatterplot(ax=ax,x=zx[:,0],y=zx[:,1],hue=labels_list,alpha=0.8,palette=label_colors_dict,s=50)
    
    ax.set(ylim=(-lim,lim))
    ax.set(xlim=(-lim,lim))
    
    if showtopic:
      ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)
   
      for indx, topic in enumerate(zphi):
        if bold_topics:
          ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13,fontweight='bold')
        else: ax.text(zphi[indx, 0], zphi[indx, 1], 'topic'+str(indx),fontsize=13)

    if hv_qwords:
      for i in range(len(query_words)):
        if (i==len(query_words)-1):
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)
        else:
          ax.text(query_words[i][0],query_words[i][1], 'X'+keywords[i] ,c='black',weight='bold',fontsize=13)

    plt.setp(g.get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(g.get_legend().get_title(), fontsize='20') # for legend title
    plt.tight_layout()
    
    if remove_legend:
      g.legend_.remove()
    if not show_axis:
      plt.axis('off')
    if save:
      plt.savefig(model_name+"_"+figname+".png", bbox_inches='tight')
    # ax.text(query_center[0],query_center[1], 'X' ,c='black',weight='bold',fontsize=20)
    ax.text(0,0, 'X' ,c='black',weight='bold',fontsize=20)
    return ax