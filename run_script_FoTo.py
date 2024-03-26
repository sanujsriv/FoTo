import os
from time import time
import argparse

parser = argparse.ArgumentParser(description='FoTo')
parser.add_argument('--data_name', type=str, default='bbc', help='bbc,searchsnippet, wos')
parser.add_argument('--dtype', type=str, default='short', help='full,short,small')
parser.add_argument('--num_topic', type=str, default='10', help='number of topic')
parser.add_argument('--num_runs', type=str, default='1', help='# run')
parser.add_argument('--queryset', type=int, default=1, help='queryset used to run the model')
parser.add_argument('--extended', type=str, default='1', help='to use extended list of keywords')
args = parser.parse_args()

# data_name = 'bbc' # bbc,searchsnippet, wos
# dtype = 'short' # full,short,small

data_name = args.data_name
# dtype = args.dtype
dtype = 'short'
num_topic =args.num_topic
num_runs = [args.num_runs]
drop = 0.2
queryset = args.queryset

# ext = args.extended
ext=1
th = 0.5
epochs ='1000'


paper="emnlp2022"
model_name = 'FoTo'
home_dir='/home/grad16/sakumar/'+paper+'/'+model_name
os.chdir(home_dir)
save_dir_no_bkp = '/home/student_no_backup/sakumar/'+paper+'/'+model_name

for r in num_runs:
  print(r)
  os.chdir(home_dir)
  # save_dir = save_dir_no_bkp+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/th_"+str(th)+"/qs_"+str(queryset)+"/run_"+str(r) # +"/ext"+str(ext)+
  save_dir = save_dir_no_bkp+"/SavedOutput/"+data_name+"/"+dtype+"/topics_"+str(num_topic)+"/ext"+str(ext)+"/qs_"+str(queryset)+"/run_"+str(r) #
  os.makedirs(save_dir,exist_ok=True)
  # os.chdir(save_dir)
  os.system("nohup python3 main.py \
   --dataset "+data_name+" --dtype "+dtype+" --num_topic "+num_topic+" --run "+str(r)+" -e "+epochs+" -qs "+str(queryset)+" -th "+str(th)+" > " \
   +save_dir+"/"+"output_"+data_name+"_"+str(num_topic)+".txt")
