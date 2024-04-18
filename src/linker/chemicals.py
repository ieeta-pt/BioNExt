from src.data import load_data
from collections import defaultdict
import json
import os

# load embeddings
import os
import numpy as np
import torch
from src.linker.utils import NORMALIZER_MODEL_MAPPINGS_REVERSED
from transformers import AutoTokenizer, AutoModel
from functools import partial, lru_cache
from tqdm import tqdm


def build_direct_lookup_function(data_dict, transform_entity):
    def _lookup(entity):
        entity = transform_entity(entity)
        return data_dict.get(entity, [])
    return _lookup
    
def run_chemicals(testset, output_file, dataset_folder, kb_folder):
    
    print("load training data and kbases for chemicals")
    training_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_train.json"))
    val_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_val_revealed.json"))
    training_data.extend(val_data)

    test_run = load_data(testset)
    
    USE_LOWER_CASE = True
    MIN_EMB_THREASHOLD = 0.9
    if USE_LOWER_CASE:
    ## LOAD the exact match files
        def transform_entity(x):
            return x.lower()
    else:
        def transform_entity(x):
            return x

    training_direct_match = defaultdict(list)
    for doc in training_data:
        for entity in doc["entities"]:
            if entity["label"] == "ChemicalEntity" or entity["label"] == "Chemical":
                training_direct_match[transform_entity(entity["text"])].append(entity["linked_id"])
    training_lookup = build_direct_lookup_function(training_direct_match, transform_entity)
    
    
    device = "cuda"

    embeddings= {}
    embeddings_id = {}
    _path = f"{kb_folder}/MeSH/"

    for file_p in filter(lambda x: x.endswith(".jsonl"), os.listdir(_path)):
        with open(os.path.join(_path, file_p)) as f:
            embeddings_id[file_p.split(".")[0]] = [x["id"] for x in map(json.loads, f)]

    for file_p in filter(lambda x:x.endswith(".npy"), os.listdir(_path)):
        _embeddings = torch.as_tensor(np.load(os.path.join(_path, file_p))).to(device)
        _embeddings = _embeddings/torch.linalg.norm(_embeddings, ord=2, axis=-1, keepdims=True)
        embeddings[file_p.split("_")[0]] = _embeddings
        

    checkpoint = NORMALIZER_MODEL_MAPPINGS_REVERSED[file_p.split("_")[-1][:-4]]
            
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    model = AutoModel.from_pretrained(checkpoint).to(device)
    
    
    @lru_cache(maxsize=1_000_000)
    def embedding_lookup_function(entity, threshold=MIN_EMB_THREASHOLD):
    
        entity = transform_entity(entity)
        best_matches_per_embfile = []
        
        inputs = tokenizer(entity, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model(**inputs)[0].mean(axis=1)
            embedding = embedding/torch.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
        
        for dict_key in embeddings.keys():
            
                scores = (embeddings[dict_key] @ embedding.T).squeeze()
                max_score, index_max = torch.max(scores, dim=-1)
                max_score = max_score.cpu().numpy().item()
                index_max = index_max.cpu().numpy().item()
                if max_score>threshold:
                    best_matches_per_embfile.append((embeddings_id[dict_key][index_max], max_score))
        
        if len(best_matches_per_embfile)>0:
            return [max(best_matches_per_embfile, key=lambda x:x[1])[0]]
        else:
            return []
        
    cascade_order = [
        training_lookup,
        partial(embedding_lookup_function, threshold=MIN_EMB_THREASHOLD),
    ]

    for doc in tqdm(test_run):
        for entity in doc['entities']:
            if entity["label"] == "ChemicalEntity" or entity["label"] == "Chemical":

                prediction = []
                
                for lookup_fn in cascade_order:
                    if len(prediction) == 0:
                        prediction.extend(lookup_fn(entity['text']))
                    else:
                        break
                
                if len(prediction) == 0:
                    entity['linked_id'] = '-' 
                else:
                    entity['linked_id'] = prediction
                    
                    
    # Desambiguation based on the document
    for doc in test_run:
        id_entities = defaultdict(list)
        for entity in doc["entities"]:
            if entity["label"] == "ChemicalEntity" or entity["label"] == "Chemical":
                if isinstance(entity["linked_id"], list):
                    for linked_id in entity["linked_id"]:
                        id_entities[linked_id].append(entity["id"])
                        # print(id_entities)
                    
        # do majoraty voting (pick the id that has the longest list that each entity belongs too)
        for entity in doc["entities"]:
            if entity["label"] == "ChemicalEntity" or entity["label"] == "Chemical":
                if isinstance(entity["linked_id"], list):
                    
                    most_freq_id, _ = max([(linked_id, len(id_entities[linked_id])) for linked_id in entity["linked_id"]], key=lambda x:x[1])                    
                    entity["linked_id"] = most_freq_id

    test_run_data_dict = {doc["id"]:{str(ent["id"]):ent for ent in doc["entities"] if ent["label"] == "ChemicalEntity" or ent["label"] == "Chemical"} for doc in test_run}
    
    # build doc
    with open(testset) as f:
        testdata = json.load(f)
    c = 0
    for doc in testdata["documents"]:
        annotated_entities = test_run_data_dict[doc["id"]]
        for passage in doc["passages"]:
            for annotation in passage["annotations"]:
                if annotation["id"] in annotated_entities and annotated_entities[annotation["id"]]["linked_id"] != "-":
                    c +=1
                    annotation["infons"]["identifier"] = annotated_entities[annotation["id"]]["linked_id"]
    
    print("number of predicted chemicals:", c)
    with open(output_file,"w") as fOut:
        json.dump(testdata, fOut, indent=2)
        
    return output_file
        
 
