import os
import yaml
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, PretrainedConfig, BertConfig, AutoConfig

import torch
from huggingface_hub import hf_hub_download
from transformers import BertConfig
from src.extractor.model import *
from safetensors import safe_open

def load_model_and_tokenizer(model_checkpoint, revision):

    model, config = load_model(model_checkpoint, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, 
                                              revision=revision,
                                              cache_dir="../HF_CACHE")



    return model, tokenizer, config

def load_model_local(model_checkpoint, device="cuda"):
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    config = AutoConfig.from_pretrained(model_checkpoint)
    #print(config) 
    #AutoConfig.from_json_file(model_checkpoint)

    class_name = eval(config.architectures[0])
        
    model = class_name(config=config).to(device)
    #model = AutoModel.from_pretrained("microsoft-bilstm-64-42-roberta-es-ner-trainer-v5/checkpoint-2070")
    if os.path.exists(f"{model_checkpoint}/pytorch_model.bin"):
        model.load_state_dict(torch.load(f"{model_checkpoint}/pytorch_model.bin", map_location=torch.device(device)))
    elif os.path.exists(f"{model_checkpoint}/model.safetensors"):
        state_dict = {}
        with safe_open(os.path.join(model_checkpoint, "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict)
        
    return model, tokenizer, config

def load_model(model_checkpoint, revision):
    #config = BERT.from_pretrained(model_checkpoint, 
    #                                    revision=revision, 
    #                                    cache_dir="../HF_CACHE")
    
    path_to_cfg = hf_hub_download(repo_id=model_checkpoint, 
                                    revision=revision, 
                                    filename="config.json", 
                                    cache_dir="../HF_CACHE")
    
    config = BertConfig.from_json_file(path_to_cfg)

    print(config)
    if config.model_type_arch=="dense":
        model = BERTDenseCRF(config=config)
    elif config.model_type_arch=="bilstm":
        model = BERTLstmCRF(config=config)
    else:
        raise ValueError(f"Invalid model_type_arch: {config.model_type_arch}")
    
    if revision != "main":
        path_to_bin = hf_hub_download(repo_id=model_checkpoint, 
                                    revision=revision, 
                                    filename="pytorch_model.bin", 
                                    cache_dir="../HF_CACHE")
    else:
        path_to_bin = model_checkpoint+"/pytorch_model.bin"
    
    print("LOAD THE WEIGHTS")
    model.load_state_dict(torch.load(path_to_bin, map_location=torch.device('cpu')))
    
    return model, config


def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def setup_wandb(name=None):
    assert name is not None
    os.environ["WANDB_NAME"] = name
    os.environ["WANDB_API_KEY"] = open(".api").read().strip()
    os.environ["WANDB_PROJECT"] = "BC8-BioRED-Track1-RE"
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_ENTITY"] = "bitua"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    os.environ["WANDB_DIR"]="~/WANDB_LOGS"

class RangeDict():
    
    def __init__(self):
        super().__init__()
        self.length = 0
        self.data = {}
    
    def __getitem__(self, span: tuple):
        return self.data[span[0]]
        
    def __setitem__(self, span: tuple, val):
        assert span[0]<span[1]        

        for i in range(*span): 
            self.data[i] = val
        self.length += 1

    def __len__(self):
        return self.length
    
    def span_collision(self, span) -> int:
        c = 0
        l = []
        for i in range(*span): 
            if i in self.data:
                c+=1
                l.append(self.data[i])
        return c , l  
    
    def maybe_merge_annotations(self, annotation):

        c, l = self.span_collision(((annotation["start_span"], annotation["end_span"])))
        
        # there is a collision issue?
        if c>0:
            ann_starts = list(map(lambda x:x["start_span"],l))
            ann_ends = list(map(lambda x:x["end_span"],l))
            ann_start, ann_end = min(ann_starts+[annotation["start_span"]]), max(ann_ends+[annotation["end_span"]])
            
            return ann_start, ann_end
            
        return None
        
    def get_all_annotations(self):
        ann_id_set = set()
        unique_anns = []
        for ann in self.data.values():
            if ann["id"] not in ann_id_set:
                ann_id_set.add(ann["id"])
                unique_anns.append(ann)
                
        
        return unique_anns   

    
    def __contains__(self, span: tuple) -> bool:
        #print("SPAN",span)
        return span[0] in self.data and span[1]-1 in self.data and self.data[span[0]] == self.data[span[1]-1]
    
def _load_flat_config(path):
    assert path is not None, "`path` cannot be none"
    with open(path) as fp:
        config = yaml.safe_load(fp)
    
    return _flatten(config)
        
    
def _flatten(d):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(_flatten(v).items())
            elif isinstance(v, list):
                for x in v:
                    items.extend(_flatten(x).items())
            else:
                try:
                    items.append((k, eval(v)))
                except (NameError, TypeError):
                    items.append((k, v))
    else:
        raise ValueError(f"Found leaf value ({repr(d)}) that is not a dictionary. Please convert it to a dictionary.")
    return dict(items)

def create_config(base_config_path="bert_trainer_config.yaml", **update_config):
    #assert isinstance(update_config, dict), "update_config config must be a dictionary."
    print(f"Combining values supplied as `keywords arguments` with base config from {base_config_path}" if update_config else f"Using base config from {base_config_path}")
    
    base_config = _load_flat_config(base_config_path)
    joint_config = base_config | update_config
    return TrainingArguments(**joint_config)

