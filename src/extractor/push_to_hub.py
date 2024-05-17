from model.modeling_bionextextractor import BioNExtExtractorModel
from model.configuration_bionextextractor import BioNExtExtractorConfig
import argparse 

from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="")
parser.add_argument("token", type=str, default=None)

args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained("../../trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200/config.json")


config = BioNExtExtractorConfig.from_json_file("../../trained_models/extractor/biolinkbert-large-full-mha-both-3-32456-20-mask-False/checkpoint-17340/config.json")
config.architectures[0] = "BioNExtExtractorModel"
config.vocab_size = config.update_vocab
config.update_vocab = config.update_vocab

model = BioNExtExtractorModel(config)
model.training_mode()

from safetensors import safe_open
import os
state_dict = {}
with safe_open(os.path.join("../../trained_models/extractor/biolinkbert-large-full-mha-both-3-32456-20-mask-False/checkpoint-17340", "model.safetensors"), framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)
print(model.load_state_dict(state_dict))

config.register_for_auto_class()
model.register_for_auto_class("AutoModel")

model.push_to_hub('IEETA/BioNExt-Extractor', token=args.token)
config.push_to_hub('IEETA/BioNExt-Extractor', token=args.token)
tokenizer.push_to_hub('IEETA/BioNExt-Extractor', token=args.token)

