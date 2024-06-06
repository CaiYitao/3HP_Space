
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import pandas as pd
# from utils import *
from shortest_path_bipartite_ import *
import sys
# sys.path.append("/home/talax/xtof/local/Mod/lib64")
sys.path.append("/home/mescalin/yitao/Documents/Code/CRN_IMA/OpenBioMed-main")
import subprocess
# from mod import *
# from open_biomed import *
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


def main():
    
    # model =  transformers.AutoModelForCausalLM.from_pretrained("PharMolix/BioMedGPT-LM-7B",low_cpu_mem_usage=True)
    model = LlamaForCausalLM.from_pretrained("PharMolix/BioMedGPT-LM-7B", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained("PharMolix/BioMedGPT-LM-7B")
    model = model.to("cuda")
    tokenizer = tokenizer.to("cuda")
    # pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer)
    input = tokenizer("how to desgin a drug",truncation=True, return_tensors="pt")
    print(input)
    print(model)
    output = model.generate(inputs=input.input_ids,max_new_tokens = 256,early_stopping=True,do_sample=True)
    print(output)
    tokenizer.decode(output[0],skip_special_tokens=True)
    # pipe("how to desgin a drug",max_new_tokens=4090)

# class Actor(nn.module):
#     def 


if __name__ == "__main__":

    print("CUDA is available:",torch.cuda.is_available())   
    main()