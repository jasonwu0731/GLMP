import os
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False)
parser.add_argument('-t','--task', help='Task Number', required=False, default="")
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False)
parser.add_argument('-dr','--drop', help='Drop Out', required=False)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-l','--layer', help='Layer Number', required=False)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)

parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs','--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-abg','--ablationG', help='ablation global memory pointer', type=int, required=False, default=0)
parser.add_argument('-abh','--ablationH', help='ablation context embedding', type=int, required=False, default=0)
parser.add_argument('-rec','--record', help='use record function during inference', type=int, required=False, default=0)
# parser.add_argument('-beam','--beam_search', help='use beam_search during inference, default is greedy search', type=int, required=False, default=0)
# parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)

args = vars(parser.parse_args())
print(str(args))
print("USE_CUDA: "+str(USE_CUDA))

LIMIT = int(args["limit"]) 
MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4

if args["ablationG"]: args["addName"] += "ABG"
if args["ablationH"]: args["addName"] += "ABH"
