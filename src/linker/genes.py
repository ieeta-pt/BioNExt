from src.data import load_data
from collections import defaultdict
import json
import os
import pickle
from tqdm import tqdm
from functools import partial, lru_cache


# load embeddings
import os
import numpy as np
import torch
from src.linker.utils import NORMALIZER_MODEL_MAPPINGS_REVERSED
from transformers import AutoTokenizer, AutoModel

def build_direct_lookup_function(data_dict, transform_entity):
    def _lookup(entity):
        entity = transform_entity(entity)
        return data_dict.get(entity, [])
    return _lookup
    
def run_genes(testset, output_file, dataset_folder, kb_folder):
    
    print("load training data and kbases for genes")
    training_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_train.json"))
    val_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_val_revealed.json"))
    training_data.extend(val_data)
    
    test_run = load_data(testset)
    
    with open(f"{kb_folder}/NCBI-Gene/genes_with_tax.pickle", "rb") as handle:
        genes = pickle.load(handle)
    
    train_data_dict = defaultdict(lambda: defaultdict(lambda: set()))
    for doc in training_data:
        organism_ids = [] 
        for entity in doc['entities']:
            if entity['label']=='OrganismTaxon' or entity["label"] == "Organism":
                organism_ids.append((entity['linked_id'], entity['start_span']))
        #add default for human
        if len(organism_ids) == 0:
            organism_ids.append(('9606', 0))

        for entity in doc['entities']:
            if entity['label']=='GeneOrGeneProduct' or entity['label']=='Gene':
                distance = 10_000_000 
                for org in organism_ids:
                    if abs(entity['start_span']-org[1]) < distance and org[0] in genes.keys():
                        distance=abs(entity['start_span']-org[1])
                        nearest_org=org[0]

                train_data_dict[nearest_org][entity['text'].lower()].add(entity['linked_id']) 
                
    backup_gene = defaultdict(lambda: set())
    for tax, gene_map in tqdm(genes.items()):
        for w, gens in gene_map.items():
            backup_gene[w].update(gens)
    
    device = "cuda"

    embeddings= {}
    embeddings_id = {}
    _path = f"{kb_folder}/NCBI-Gene/embeddings"

    for file_p in filter(lambda x: x.endswith(".jsonl"), os.listdir(_path)):
        with open(os.path.join(_path, file_p)) as f:
            embeddings_id[file_p.split("_")[0]] = [x["id"] for x in map(json.loads, f)]

    for file_p in filter(lambda x:x.endswith(".npy"), os.listdir(_path)):
        _embeddings = torch.as_tensor(np.load(os.path.join(_path, file_p))).to(device)
        _embeddings = _embeddings/torch.linalg.norm(_embeddings, ord=2, axis=-1, keepdims=True)
        embeddings[file_p.split("__")[0]] = _embeddings
    
    print("Loaded genes emb files", embeddings.keys())

    checkpoint = NORMALIZER_MODEL_MAPPINGS_REVERSED[file_p.split("_")[-1][:-4]]
            
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    model = AutoModel.from_pretrained(checkpoint).to(device)
    
    MIN_EMB_THREASHOLD = 0.9
    
    @lru_cache(maxsize=1_000_000)
    def get_code_wemb(entity_text, nearest_org):
        inputs = tokenizer(entity_text.lower(), return_tensors="pt").to(device)
        gene_code = "-"
        with torch.no_grad():
            embedding = model(**inputs)[0].mean(axis=1)
            embedding = embedding/torch.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
            scores = (embeddings[nearest_org] @ embedding.T).squeeze()
            max_score, index_max = torch.max(scores, dim=-1)
            max_score = max_score.cpu().numpy().item()
            index_max = index_max.cpu().numpy().item()
            #print(max_score)
            if max_score>MIN_EMB_THREASHOLD:
                gene_code = embeddings_id[nearest_org][index_max]
            
                #print("Embedding code not found")
            del embedding
            del scores
            del inputs
        return gene_code
    
    counter_total = 0
    
    for doc in tqdm(test_run):
        
        #builds (org_id, start) for a doc    
        organism_ids = [] 
        for entity in doc['entities']:
            if entity['label']=='OrganismTaxon' or entity['label']=='Organism':
                organism_ids.append((entity['linked_id'], entity['start_span']))
        #add default for human, if no organism is found
        if len(organism_ids) == 0:
            organism_ids.append(('9606', 0))
        
        for entity in doc['entities']:
            if entity['label']=='GeneOrGeneProduct' or entity['label']=='Gene':
                counter_total+=1


                #looks up nearest org
                distance = 10_000_000 
                for org in organism_ids:
                    if abs(entity['start_span']-org[1]) < distance and org[0] in genes.keys():
                        distance=abs(entity['start_span']-org[1])
                        nearest_org=org[0]

                entity["pred_tax"] = nearest_org
                prediction = []
                
                # from training
                if len(prediction) == 0:
                    if nearest_org in train_data_dict:
                        if entity['text'].lower() in dict(train_data_dict[nearest_org]):
                            gene_id=dict(train_data_dict[nearest_org])[entity['text'].lower()]
                            # gene_id = list(gene_id)[0]
                            prediction.extend(list(gene_id))
                            
                if len(prediction) == 0:
                    if entity['text'].lower() in genes[nearest_org].keys():
                        gene_id=genes[nearest_org][entity['text'].lower()]
                        prediction.extend(list(set(gene_id)))
                        
                if len(prediction) == 0:
                    if nearest_org in embeddings:
                        #print("embedding?")
  
                        prediction.append(get_code_wemb(entity["text"], nearest_org))
                       
                        
                if len(prediction) == 0:
                    if entity['text'].lower() in backup_gene.keys():
                        gene_id=backup_gene[entity['text'].lower()]
                        prediction.extend(list(gene_id))
                if len(prediction) == 0:
                    entity['linked_id'] = '-' # ""
                else:
                    entity['linked_id'] = prediction
                            
    # Desambiguation based on the document
    for doc in test_run:
        id_entities = defaultdict(list)
        for entity in doc["entities"]:
            if entity["label"] == "GeneOrGeneProduct" or entity["label"] == "Gene":
                if isinstance(entity["linked_id"], list):
                    for linked_id in entity["linked_id"]:
                        id_entities[linked_id].append(entity["id"])
                        # print(id_entities)
                    
        # do majoraty voting (pick the id that has the longest list that each entity belongs too)
        for entity in doc["entities"]:
            if entity["label"] == "GeneOrGeneProduct" or entity["label"] == "Gene":
                if isinstance(entity["linked_id"], list):
                    
                    most_freq_id, _ = max([(linked_id, len(id_entities[linked_id])) for linked_id in entity["linked_id"]], key=lambda x:x[1])                    
                    entity["linked_id"] = most_freq_id

    test_run_data_dict = {doc["id"]:{str(ent["id"]):ent for ent in doc["entities"] if ent["label"] == "GeneOrGeneProduct" or ent["label"] == "Gene"} for doc in test_run}
    
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
    
    print("number of predicted gene:", c)
    with open(output_file,"w") as fOut:
        json.dump(testdata, fOut, indent=2)
                
    return output_file