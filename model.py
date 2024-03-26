from torch import Tensor
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn import Parameter
import gc

torch.cuda.empty_cache()
gc.collect()

def sgn(x) : return x/abs(x)
def ranking_0_or_1(x): return torch.nan_to_num(( (sgn(x) + 1) / 2).ceil()) 

def one_or_zero(x): return x / (abs(x-1) + 1) # 0 if x = 0 else 1 for x > 1

def gaussian(alpha): return -0.5*alpha 
def inverse_quadratic(alpha): return -torch.log(torch.ones_like(alpha) + alpha)
def toT(a): return torch.tensor(a)

class FoTo(nn.Module):
    def __init__(self, num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, drop_rate, variance_x, bs, 
                 embedding_words,embedding_keywords, word_emb_size,num_keyword,activation, distance="gaussian"):
      
        super(FoTo, self).__init__()

        self.num_input, self.num_coordinate, self.num_topic, self.variance_x, self.bs ,self.word_emb_size, self.num_keyword \
            = num_input, num_coordinate, num_topic, variance_x, bs,word_emb_size,num_keyword

        self.embedding_words = embedding_words
        self.embeddings_keys = embedding_keywords
        self.activation = activation
        
        # encoder
        self.en1_fc     = nn.Linear(num_input, en1_units_x)             # V -> 100 #nxV->(vxh1)->nxh1; 
        self.en2_fc     = nn.Linear(en1_units_x, en2_units_x)             # 100  -> 100
        self.en2_drop   = nn.Dropout(drop_rate)
        self.mean_fc    = nn.Linear(en2_units_x, num_coordinate)        # 100  -> 2
        self.mean_bn    = nn.BatchNorm1d(num_coordinate)              # bn for mean
        self.logvar_fc  = nn.Linear(en2_units_x, num_coordinate)        # 100  -> 2
        self.logvar_bn  = nn.BatchNorm1d(num_coordinate)              # bn for logvar
        
        # # mapping network for topic embeddings
        self.mu1_fc     = nn.Linear(2, 100) 
        self.mu2_fc     = nn.Linear(100, 100)
        self.mu_fc = nn.Linear(100, 300)
        
        # mapping network for topics ( embeddings --> coordinates)
        self.c1_fc     = nn.Linear(300, 100) 
        self.c2_fc     = nn.Linear(100, 100)
        self.c_fc = nn.Linear(100, 2)


        self.mu_z = nn.Parameter(torch.Tensor(self.num_topic, self.word_emb_size))   
        self.beta_bias = nn.Parameter(torch.Tensor(self.num_topic,self.num_input))

        if distance=="gaussian": self.basis_func = gaussian
        if distance=="inverse_quadratic": self.basis_func = inverse_quadratic
        self.init_parameters()

        # decoder layer
        self.decoder    = nn.Linear(self.num_topic, self.num_input)  
        self.decoder_bn = nn.BatchNorm1d(self.num_topic)
                   
        # decoder batch norm
        self.decoder_phi_bn = nn.BatchNorm1d(num_coordinate)                      
        self.decoder_x_bn = nn.BatchNorm1d(num_coordinate)
        self.decoder_q_bn = nn.BatchNorm1d(num_coordinate)
 
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_coordinate).fill_(0)
        prior_var    = torch.Tensor(1, num_coordinate).fill_(variance_x)
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        self.prior_var  = nn.Parameter(prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(prior_var.log(), requires_grad=False)
        
    def init_parameters(self):
        nn.init.normal_(self.mu_z, 0, 0.01)
        nn.init.normal_(self.beta_bias, 0, 0.01)

    def get_beta(self):
        return self.beta
          
    def get_activation(self, activation,layer):
      activation = activation.lower()
      if activation == 'relu':
          layer =  F.relu(layer)
      elif activation == 'softplus':
          layer =  F.softplus(layer)
      elif activation == 'sigmoid':
          layer =  F.sigmoid(layer)
      elif activation == 'leaky_relu':
          layer = F.leaky_relu(layer)
      else:
          layer = F.relu(layer)
      return layer 
    
    def encode(self, input_,input_keywords_as_docs,first_batch):
        if first_batch:input_w_keys = torch.cat([input_, input_keywords_as_docs], dim=0)
        else:input_w_keys = input_
        
        input_ = input_w_keys

        N, *_ = input_.size()

        en1 = self.get_activation(self.activation,self.en1_fc(input_))                         
        en2 = self.get_activation(self.activation,self.en2_fc(en1)) 
        # en1 = F.relu(self.en1_fc(input_))          
        # en2 = F.relu(self.en2_fc(en1))                    
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))   # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))   # posterior log variance
        posterior_var    = posterior_logvar.exp()
        return posterior_mean, posterior_logvar, posterior_var
    
    def take_sample(self, input_,input_keywords_as_docs,first_batch, posterior_mean, posterior_var, prior_var):
        if first_batch: input_w_keys = torch.cat([input_, input_keywords_as_docs], dim=0)
        else: input_w_keys = input_

        input_ = input_w_keys

        eps = input_.data.new().resize_as_(posterior_mean.data).normal_(std=1) # noise
        z = posterior_mean + posterior_var.sqrt() * eps           # reparameterization
        return z

    def get_theta(self,first_batch,input_keywords_as_docs): 
      # topic mapping
      
      c1_nn = self.get_activation(self.activation,self.c1_fc(self.mu_z))                         
      c2_nn = self.get_activation(self.activation,self.c2_fc(c1_nn)) 
      # c1_nn = F.relu((self.c1_fc(self.mu_z)))                           
      # c2_nn = F.relu(self.c2_fc(c1_nn))
      self.topics = self.c_fc(c2_nn)

      if first_batch:
        self.query_center = self.decoder_x_bn(self.z)[self.z.shape[0]-input_keywords_as_docs.shape[0]:,:]
        self.q = self.query_center
      else: self.q = self.q.data         # self.q = self.query_center

      N, *_ = self.z.size()
      zx = self.decoder_x_bn(self.z).view(N, 1, self.num_coordinate) # Nx1xX
      # zx = self.z.view(N, 1, self.num_coordinate)
      
      zc = self.decoder_phi_bn(self.topics)
      # zc = self.topics

      doc_topic_size = (N, self.num_topic, self.num_coordinate) # NxTx2
   
      x = zx.expand(doc_topic_size)
      c = zc.view(1, self.num_topic, self.num_coordinate).expand(doc_topic_size)

      d_doc_topic = (x-c).pow(2).sum(-1)
    
      distances_doc_topic = self.basis_func(d_doc_topic)

      theta = torch.exp(distances_doc_topic - torch.logsumexp(distances_doc_topic, dim=-1, keepdim=True))
      return theta ,zx, zc, N
    
    
    def decode(self,first_batch,input_keywords_as_docs):
      theta, zx, zc, N = self.get_theta(first_batch,input_keywords_as_docs)

      self.beta = F.softmax(torch.mm(self.mu_z, self.embedding_words.T) + self.beta_bias,dim=-1) #
      # self.beta = F.softmax(self.decoder_bn(torch.mm(self.mu_z,self.embedding_words.T).T).T ,dim=-1)

      recon_v_relv = torch.mm(theta,self.beta)

      return recon_v_relv, zx,zc, theta
    
    def forward(self, input_,relevant_scores,relv_scores_k,input_keywords_as_docs,first_batch,compute_loss=False):  
        posterior_mean, posterior_logvar, posterior_var = self.encode(input_,input_keywords_as_docs,first_batch)
        z = self.take_sample(input_,input_keywords_as_docs,first_batch, posterior_mean, posterior_var, self.variance_x)
        self.z = z
        recon_v_relv, zx, zphi, theta = self.decode(first_batch,input_keywords_as_docs)
        
        if compute_loss:
            return recon_v_relv,zx, self.loss(input_, input_keywords_as_docs,first_batch, recon_v_relv, posterior_mean, posterior_logvar, posterior_var, relevant_scores,relv_scores_k,zx,zphi)
        else: return z, recon_v_relv, zx,zphi, theta
 
    def KLD(self, posterior_mean,posterior_logvar,posterior_var):
        N = posterior_mean.shape[0]
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
 
        var_division    = posterior_var  / prior_var 
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        
        xKLD = 0.5 * ((var_division + diff_term + logvar_division).sum(-1) - self.num_coordinate) 
        return xKLD
    
    def loss(self, input_, input_keywords_as_docs,first_batch, recon_v, posterior_mean, posterior_logvar, posterior_var, relv_scores,relv_scores_k,zx,zphi):
        N = posterior_mean.shape[0]
        if first_batch:
          input_w_keys = torch.cat([input_, input_keywords_as_docs], dim=0)
          N = N - input_keywords_as_docs.shape[0]
          zx = zx[:input_.shape[0],:]
          
        else:
          input_w_keys = input_
        
        smoothen = 1e-12 # 'error/term-smoothening constant'
        NL1 = - (input_w_keys * (recon_v+1e-20).log()).sum(-1)
        NL1 = NL1.sum()
        
        KLD = self.KLD(posterior_mean,posterior_logvar,posterior_var).sum()

        doc_query_size = (N, self.num_keyword, self.num_coordinate) # NxKx2  
        x_q = zx.expand(doc_query_size)
        q_x = self.q.view(1, self.num_keyword, self.num_coordinate).expand(doc_query_size)

        eucl_sq_dist = (x_q - q_x).pow(2).sum(-1)
        # dist_doc_query_k = torch.exp(-0.5 * eucl_sq_dist)
        dist_doc_query = (torch.exp(-0.5 * eucl_sq_dist))


        # ## for all aspects
        # dist_A_q = dist_doc_query.unsqueeze(-1).expand(N,N)
        # dist_B_q  =  dist_A_q.T + smoothen
        # # if len(relv_scores.squeeze().shape)==1:
        # #   relv_A = relv_scores.squeeze().T.unsqueeze(-1).expand(N,N) 
        # # else:
        # relv_A = relv_scores.expand(N,N) 
        # relv_B = relv_A.T
        # ###

        ### for individual aspects
        dist_A_q = dist_doc_query.transpose(-2,-1).unsqueeze(-1).expand(self.num_keyword,N,N)
        dist_B_q  =  dist_A_q.transpose(-2,-1)  + smoothen
        if len(relv_scores.squeeze().shape)==1:
          relv_A = relv_scores.squeeze().T.unsqueeze(-1).expand(self.num_keyword,N,N) 
        else:
          relv_A = relv_scores.squeeze().transpose(-2,-1).unsqueeze(-1).expand(self.num_keyword,N,N) 
        relv_B = relv_A.transpose(-2,-1)
        ###
        
        ### for ranking aspects
        # dist_A_q_k = dist_doc_query_k.unsqueeze(-1).expand(N,self.num_keyword,self.num_keyword) 
        # dist_B_q_k  =  dist_A_q_k.transpose(-2,-1) + smoothen
        # relv_A_k = relv_scores_k.unsqueeze(-1).expand(N,self.num_keyword,self.num_keyword) 
        # relv_B_k = relv_A_k.transpose(-2,-1)
        ###
        
        B_minus_A= (dist_A_q/dist_B_q)#(dist_A_q-dist_B_q)
        # B_minus_A_k = (dist_A_q_k/dist_B_q_k)#(dist_A_q_k-dist_B_q_k)
        # print('relvA-B',torch.isnan(relv_A - relv_B).any())
        ranking = ranking_0_or_1(relv_A - relv_B)
        
        # ranking_k = ranking_0_or_1(relv_A_k - relv_B_k)

        NL2 =  - (ranking * (B_minus_A.sigmoid() + smoothen).log()).sum() 
        # NL3 =  - (ranking_k * (B_minus_A_k.sigmoid() + smoothen).log()).sum()
        NL3 = toT(0.0)
        NL = NL1 + NL2+ NL3
        
        loss = NL+ KLD
        return loss/N, NL,NL1,NL2,NL3, KLD