import os
import yaml
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, PretrainedConfig, BertConfig
from src.tagger.BERT_DENSE_CRF import BERTDenseCRF
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open



def load_model_and_tokenizer_locally(model_checkpoint):

    model, config = load_model_locally(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, 
                                              cache_dir="../HF_CACHE")



    return model, tokenizer, config

def load_model_locally(model_checkpoint):
    
    config = BertConfig.from_json_file(f"{model_checkpoint}/config.json")
    
    if config.model_type_arch=="dense":
        model = BERTDenseCRF(config=config)
    elif config.model_type_arch=="bilstm":
        model = BERTLstmCRF(config=config)
    else:
        raise ValueError(f"Invalid model_type_arch: {config.model_type_arch}")

    #path_to_bin = model_checkpoint+"/pytorch_model.bin"
    state_dict = {}
    with safe_open(os.path.join(model_checkpoint, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    
    return model, config


def load_model_and_tokenizer(model_checkpoint, revision):

    model, config = load_model(model_checkpoint, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, 
                                              revision=revision,
                                              cache_dir="../HF_CACHE")



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




def setup_wandb(name=None, project_name="BioCreativeVIII-Track1-Journal-NER-V3"):
    assert name is not None
    os.environ["WANDB_NAME"] = name
    os.environ["WANDB_API_KEY"] = open(".api").read().strip()
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_ENTITY"] = "bitua"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    os.environ["WANDB_DIR"]="~/WANDB_LOGS"


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

