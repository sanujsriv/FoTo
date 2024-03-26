import torch.optim as optim 
import torch
import numpy as np 

def check_nan_parameters(model):
    nan_parameters = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_parameters.append(name)
    return nan_parameters

def train(model,tensor_train_w,train_label,args,all_indices,doc_contains_anykey_ext,keywords_as_docs,ranking_q_for_all_doc,device):
  
  learning_rate = args.learning_rate
  beta1 = 0.99
  beta2 = 0.999
  epochs = args.epochs
  
  kld_arr,recon_arr,neg_log_rscore_arr = [],[],[]
  model.to(device)

  optimizer = optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
  doc_c_key  = doc_contains_anykey_ext.int()

  for epoch in range(epochs):

      loss_u_epoch = 0.0 ## NL loss
      loss_KLD = 0.0  ## KL loss
      loss_epoch = 0.0 ## Loss per batch 
      loss_NL1_epoch = 0.0
      loss_NL2_epoch = 0.0
      loss_NL3_epoch = 0.0
      
      model.train()
      zx_l = []
      label_l = []
      first_batch = True
      for batch_ndx in all_indices:
        # input_w = tensor_train_w[batch_ndx].to(device)
        input_w = torch.tensor(tensor_train_w[batch_ndx]).float().to(device)
        input_keywords_as_docs = torch.from_numpy(keywords_as_docs).float().to(device)
        
        relv_scores = doc_c_key[batch_ndx].to(device).unsqueeze(-1)
        relv_scores_k = ranking_q_for_all_doc[batch_ndx].to(device)

        labels = train_label[batch_ndx]
        
        recon_v, zx, (loss, loss_u,loss_NL1,loss_NL2,loss_NL3, xkl_loss) = model(input_w,relv_scores,relv_scores_k,input_keywords_as_docs,first_batch,compute_loss=True)
        # zx_l.extend(zx.data.detach().cpu().numpy())
        # label_l.extend(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        nan_params = check_nan_parameters(model)

        # Print the NaN parameters
        # if nan_params:
        #     print("NaN parameters found:")
        #     for name in nan_params:
        #         print(name)
        # else:
        #     print("No NaN parameters found.")

        loss_epoch += loss.item()
        loss_u_epoch += loss_u.item()
        loss_KLD += xkl_loss.item()
        loss_NL1_epoch += loss_NL1.item()
        loss_NL2_epoch += loss_NL2.item()
        loss_NL3_epoch += loss_NL3.item()
        first_batch = False
        
        current_model = model 
      kld_arr.append(loss_KLD)
      recon_arr.append(loss_u_epoch)
      if epoch % 10 == 0:
          # model_params_stack_TF = torch.stack([torch.isnan(p).any() for p in model.parameters()])
          # is_nan = model_params_stack_TF#.any()
          # print('model params', is_nan)
          # all_model_params = [param for param in model.parameters()]
          # print(all_model_params[0])

          print('Epoch -> {} , loss -> {}'.format(epoch,loss_epoch))
          print('recon_loss==> {} || NL1==> {} || NL2==> {} || NL3==> {}|| KLD==> {}'.format(loss_u_epoch,loss_NL1_epoch,loss_NL2_epoch,loss_NL3_epoch, loss_KLD))
          # if is_nan == True:
          #   exit(0)
          # plot_fig(np.array(zx_l),label_l,model.decoder_phi_bn(model.centres).data.cpu().numpy(),10.0,'No')
  return current_model

def test(model,args,all_indices,tensor_train_w,train_label,doc_contains_anykey_ext,keywords_as_docs,ranking_q_for_all_doc,device):
  
  num_coordinate = args.num_coordinate
  model.eval()
  x_list = []
  labels_list = []
  doc_ids = []

  first_batch = True
  doc_c_key  = doc_contains_anykey_ext.int()

  with torch.no_grad():
      for batch_ndx in all_indices:

          input_w = torch.tensor(tensor_train_w[batch_ndx]).float().to(device) 
          labels = train_label[batch_ndx]        
          input_keywords_as_docs = torch.from_numpy(keywords_as_docs).float().to(device)
          relv_scores = doc_c_key[batch_ndx].to(device).unsqueeze(-1)
          relv_scores_k = ranking_q_for_all_doc[batch_ndx].to(device)

          z, recon_v, zx, zphi, zx_phi = model(input_w,relv_scores,relv_scores_k,input_keywords_as_docs,first_batch,compute_loss=False)

          if first_batch:
            zx = zx.view(-1, num_coordinate).data.detach().cpu().numpy()[:input_w.shape[0],:]
          else: 
            zx = zx.view(-1, num_coordinate).data.detach().cpu().numpy()
          
          labels_list.extend(labels)
          x_list.extend(zx)
          doc_ids = np.append(doc_ids,batch_ndx.numpy().astype(int))
    
          if first_batch:
            query_center = model.q.data.cpu().numpy()
          first_batch =  False

      doc_ids = doc_ids.astype(int)
      x_list = np.array(x_list)
      labels_list = np.array(labels_list)
      beta = model.get_beta().data.cpu().numpy()
      zphi = zphi.data.cpu().numpy()
    
      # query_center = model.q.data.cpu().numpy()
      ir_query_center = torch.zeros(1,2)#model.ir_query_center.data.cpu().numpy()
      assert len(labels_list) == len(x_list)
  return x_list,labels_list,zphi,doc_ids,beta,query_center,ir_query_center