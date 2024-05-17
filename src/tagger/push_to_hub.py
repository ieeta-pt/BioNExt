from model.modeling_bionexttagger import BioNExtTaggerModel
from model.configuration_bionexttager import BioNExtTaggerConfig
import argparse 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../../trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200/config.json")

parser = argparse.ArgumentParser(description="")
parser.add_argument("token", type=str, default=None)

args = parser.parse_args()


config = BioNExtTaggerConfig.from_json_file("../../trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200/config.json")
config.architectures[0] = "BioNExtTaggerModel"

model = BioNExtTaggerModel(config)


from safetensors import safe_open
import os
state_dict = {}
with safe_open(os.path.join("../../trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200/", "model.safetensors"), framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
print(model.load_state_dict(state_dict))

config.register_for_auto_class()
model.register_for_auto_class("AutoModel")

model.push_to_hub('IEETA/BioNExt-Tagger', token=args.token)
config.push_to_hub('IEETA/BioNExt-Tagger', token=args.token)
tokenizer.push_to_hub('IEETA/BioNExt-Tagger', token=args.token)

