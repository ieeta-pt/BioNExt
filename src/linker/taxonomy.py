
from src.data import load_data
from collections import defaultdict
import json
import os


# load embeddings
import os
import numpy as np

from tqdm import tqdm

def load_taxonomy(input = '/home/biocreative/BioCreativeVIII_Track1/knowledge-resources/NCBI-Taxonomy/names.dmp'):
    # og code used /home/biocreative/BioCreativeVIII_Track1/knowledge-resources/NCBI-Taxonomy/names.dmp
    #names = []
    #unique_names = []


    #with open(input) as f:
    #    for line in f.readlines():
            # line = f.readline()
    #        fields = line.strip().split('\t|\t')
    #        fields[-1] = fields[-1].rstrip('\t|\n')
    #        tax_id, name_txt, unique_name, name_class = fields
    #        names.append({'id': tax_id, 'text': name_txt.lower(), 'class':name_class})
    #        if unique_name != '':
    #            unique_names.append({'id': tax_id, 'text': unique_name.lower(), 'class':name_class})

    #    return names, unique_names

    with open(input) as f:
        names = [x for x in map(json.loads, f)]

    return names, None

def build_direct_lookup_function(data_dict, transform_entity):
    def _lookup(entity):
        entity = transform_entity(entity)
        return data_dict.get(entity, [])
    return _lookup



def run_taxonomy(testset, output_file, dataset_folder, kb_folder):
        
    print("load training data and kbases for taxonomy")
    training_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_train.json"))
    val_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_val_revealed.json"))
    training_data.extend(val_data)
    
    test_run = load_data(testset)
    training_direct_match = {entity["text"]:entity["linked_id"] for doc in training_data for entity in doc["entities"] if entity["label"] == "OrganismTaxon" or entity["label"] == "Organism"}
    names, unique_names = load_taxonomy(os.path.join(kb_folder, "NCBI-Taxonomy","names.jsonl"))
    
    taxonomy_dict = defaultdict(list)
    for x in names:
        taxonomy_dict[x["text"]].append(x["id"])
    
    manual_correction_to_entries = {
        "3052230":"11103"
    }
    
    
    
    # Entry Merged. Taxid 11103 was merged into taxid 3052230.
    for doc in tqdm(test_run):
        for entity in doc["entities"]:
            if entity["label"] == "OrganismTaxon" or entity["label"] == "Organism":
                if entity["text"] in training_direct_match:
                    entity["linked_id"] = [training_direct_match[entity["text"]]]
                elif entity["text"].lower() in taxonomy_dict:
                    entity["linked_id"] = taxonomy_dict[entity["text"].lower()]
                
    # desambiguate
    for doc in test_run:
        id_entities = defaultdict(list)
        for entity in doc["entities"]:
            if entity["label"] == "OrganismTaxon" or entity["label"] == "Organism":
                if isinstance(entity["linked_id"], list):
                    for linked_id in entity["linked_id"]:
                        id_entities[linked_id].append(entity["id"])
                    
        # do majoraty voting (pick the id that has the longest list that each entity belongs too)
        for entity in doc["entities"]:
            if entity["label"] == "OrganismTaxon" or entity["label"] == "Organism":
                if isinstance(entity["linked_id"], list):
                    most_freq_id, _ = max([(linked_id, len(id_entities[linked_id]))for linked_id in entity["linked_id"]], key=lambda x:x[1])
                    if most_freq_id in manual_correction_to_entries:
                        most_freq_id = manual_correction_to_entries[most_freq_id]
                        
                    entity["linked_id"] = most_freq_id
                    
    val_inference_data_dict = {doc["id"]:{str(ent["id"]):ent for ent in doc["entities"] if ent["label"] == "OrganismTaxon" or ent["label"] == "Organism"} for doc in test_run}
    # build doc
    with open(testset) as f:
        testdata = json.load(f)
    c = 0
    for doc in testdata["documents"]:
        annotated_entities = val_inference_data_dict[doc["id"]]
        for passage in doc["passages"]:
            for annotation in passage["annotations"]:
                annotation["id"] = str(annotation["id"])
                if annotation["id"] in annotated_entities and annotated_entities[annotation["id"]]["linked_id"] != "-":
                    c +=1
                    annotation["infons"]["identifier"] = annotated_entities[annotation["id"]]["linked_id"]
    
    print("number of predicted species:", c)
    with open(output_file,"w") as fOut:
        json.dump(testdata, fOut, indent=2)
        
    return output_file