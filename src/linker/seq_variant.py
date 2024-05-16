
from src.data import load_data
from collections import defaultdict
import json
import os
import torch
from tqdm import tqdm
import re
import requests
from functools import partial, lru_cache
# load embeddings
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import Dataset

import diskcache as dc

cache = dc.Cache('cache_llm_v5') #Cache(CACHE_TYPE='filesystem', CACHE_DIR='cache_llm')


def normalize_emb(emb):
    return emb/torch.linalg.norm(emb,ord=2, axis=-1, keepdims=True)

def batch_generator(dataset, batch_size):

    batch = []
    
    for text, normalize_text, nearest_gene in dataset:
        batch.append(text) 
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  
        yield batch


def build_get_embeddings(tokenizer, model, device, normalize=False):
    
    @lru_cache(maxsize=1_000_000)
    def get_embeddings(text):
        
        with torch.no_grad():
            
            toks_cuda = tokenizer.batch_encode_plus(text, 
                                                    padding="max_length", 
                                                    max_length=25, 
                                                    truncation=True,
                                                    return_tensors="pt").to(device)

            # cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
            cls_rep = model(**toks_cuda)[0].mean(axis=1)
            if normalize:
                cls_rep = normalize_emb(cls_rep)
            return cls_rep
        
    return get_embeddings
    

codon_to_rna = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', # Alanine
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R', # Arginine
    'AAU': 'N', 'AAC': 'N', # Asparagine
    'GAU': 'D', 'GAC': 'D', # Aspartic Acid
    'UGU': 'C', 'UGC': 'C', # Cysteine
    'CAA': 'Q', 'CAG': 'Q', # Glutamine
    'GAA': 'E', 'GAG': 'E', # Glutamic Acid
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', # Glycine
    'CAU': 'H', 'CAC': 'H', # Histidine
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', # Isoleucine
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', # Leucine
    'AAA': 'K', 'AAG': 'K', # Lysine
    'AUG': 'M', # Methionine (Start Codon)
    'UUU': 'F', 'UUC': 'F', # Phenylalanine
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', # Proline
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S', # Serine
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', # Threonine
    'UGG': 'W', # Tryptophan
    'UAU': 'Y', 'UAC': 'Y', # Tyrosine
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', # Valine
    'UAA': '*', 'UGA': '*', 'UAG': '*', # Stop Codons
}

codon_to_dna = {
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', # Alanine
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R', # Arginine
    'AAT': 'N', 'AAC': 'N', # Asparagine
    'GAT': 'D', 'GAC': 'D', # Aspartic Acid
    'TGT': 'C', 'TGC': 'C', # Cysteine
    'CAA': 'Q', 'CAG': 'Q', # Glutamine
    'GAA': 'E', 'GAG': 'E', # Glutamic Acid
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', # Glycine
    'CAT': 'H', 'CAC': 'H', # Histidine
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', # Isoleucine
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', # Leucine
    'AAA': 'K', 'AAG': 'K', # Lysine
    'ATG': 'M', # Methionine (Start Codon)
    'TTT': 'F', 'TTC': 'F', # Phenylalanine
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', # Proline
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S', # Serine
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', # Threonine
    'TGG': 'W', # Tryptophan
    'TAT': 'Y', 'TAC': 'Y', # Tyrosine
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', # Valine
    'TAA': '*', 'TGA': '*', 'TAG': '*', # Stop Codons
}

codon_amino = codon_to_rna|codon_to_dna

def convert_amino_acids(text):

    for three_letter, one_letter in codon_amino.items():
        text = text.replace(three_letter, one_letter)

    return text




def get_variant(query, gene):
    url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/autocomplete/?query={query} {gene}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None
    
def load_train_val(training_schema, use_val, kb_path, dataset_path):
     #load dataset
    if use_val:
        tmvar_data = pd.read_csv(f"{kb_path}/dbSNP/tmVar3/seqvariants.tsv", sep="\t")
    else:
        tmvar_data = pd.read_csv(f"{kb_path}/dbSNP/tmVar3/seqvariants_without_bc8_biored.tsv", sep="\t")
    train_data = load_data(f'{dataset_path}/bc8_biored_task1_train.json')
    val_data = load_data(f'{dataset_path}/bc8_biored_task1_val_revealed.json')
    
    # Builds train data
    data_set_text =[]
    data_set_norm =[]
    data_set_nearest_g = []
    if training_schema == 'all' or training_schema == 'train':
        for doc in train_data:
            
            gene_ids = [] 
            for entity in doc['entities']:
                if entity['label']=='GeneOrGeneProduct' or entity["label"] == "Gene":
                    gene_ids.append((entity['linked_id'], entity['start_span']))
            
            for entity in doc['entities']:
                if entity['label']=='SequenceVariant':
                    
                    distance = 10_000_000 
                    for gene in gene_ids:
                        if abs(entity['start_span']-gene[1]) < distance:
                            distance=abs(entity['start_span']-gene[1])
                            nearest_gene=gene[0]
                    
                    if '|' in entity['linked_id']:
                        data_set_text.append(entity['text'])
                        data_set_norm.append(entity['linked_id'])
                        data_set_nearest_g.append(nearest_gene)
                        
    if training_schema == 'all' or training_schema == 'tmvar':
        for i in tmvar_data.iterrows():
            
            candidate = i[1]['identifier'].split(';')[0]
            if '|' in candidate: 
                data_set_text.append(i[1]['mention'])
                data_set_norm.append(candidate)
                #print("TMVAR", i[1]["identifier"].split(";")[1].split(":")[1])
                if "CorrespondingGene" in i[1]["identifier"]:
                    data_set_nearest_g.append(i[1]["identifier"].split(";")[1].split(":")[1])
                else:
                    data_set_nearest_g.append("?")
                
    
    
    ds_val_text =[]
    ds_val_norm =[]
    ds_val_nearest_g = []
    for doc in val_data:
        gene_ids = [] 
        for entity in doc['entities']:
            if entity['label']=='GeneOrGeneProduct' or entity["label"] == "Gene":
                gene_ids.append((entity['linked_id'], entity['start_span']))
                
        for entity in doc['entities']:
            if entity['label']=='SequenceVariant':
                distance = 10_000_000 
                for gene in gene_ids:
                    if abs(entity['start_span']-gene[1]) < distance:
                        distance=abs(entity['start_span']-gene[1])
                        nearest_gene=gene[0]
                if '|' in entity['linked_id']:
                    ds_val_text.append(entity['text'])
                    ds_val_norm.append(entity['linked_id'])
                    ds_val_nearest_g.append(nearest_gene)
    
    ds_train = Dataset.from_dict({"text": data_set_text, "norm": data_set_norm, "nearest_gene": data_set_nearest_g})
    ds_train = ds_train.shuffle(seed=42)
    ds_val = Dataset.from_dict({"text": ds_val_text, "norm": ds_val_norm, "nearest_gene": ds_val_nearest_g})

    return ds_train, ds_val

#train_data = load_data('../dataset/bc8_biored_task1_train.json')

device = "cuda"
    
checkpoint = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
model = AutoModel.from_pretrained(checkpoint).to(device)

get_embeddings = build_get_embeddings(tokenizer, model, device, True)


pattern = r"(c|p)\|SUB\|[A-Z]\|(\d+)(\|)?$"

def filter_sub_w_allele(seq):

    match = re.match(pattern, seq)
    return match and match.group() == seq 





def run_seq_variant(testset, output_file, dataset_folder, kb_folder, llm_call):
    
    def build_ollama_predictor(embeddings, data_set_train, thearshold=0.6):

        @cache.memoize()
        def llm_predict(text, gene):
            
            # find top 5 embeddings
            target_emb = get_embeddings((text,))

            values, indice = torch.topk((embeddings @ target_emb.T).squeeze(), k=50)
            
            #prompt = f"""As a biomedical expert, your role involves accurately associating sequence variances with their corresponding identifiers. You are provided with five examples to follow. Complete the last entry of the gene sequence variant.
    #
    #Example format:
    #
    #"""
            prompt = """As a biomedical expert, your role involves accurately associating sequence variances with their corresponding identifiers, following tmVar annotation guidelines. 

    Key Annotation Rules:

    Substitution: <Sequence type>|SUB|<wild type>|<mutation position>|<mutant>
    Deletion: <Sequence type>|DEL|<mutation position>|<mutant>
    Insertion: <Sequence type>|INS|<mutation position>|<mutant>
    Insertion + Deletion (indel/delins): <Sequence type>|INDEL|<mutation position>|<mutant>
    Duplication: <Sequence type>|DUP|<mutation position>|<mutant>|<duplication times>
    Frame shift: <Sequence type>|FS|<wild type>|<mutation position>|<mutant>|<frame shift position>
    Sequence types include DNA (c), RNA (r), Genome (g), Protein (p), and Mitochondrial (m) sequences.

    Examples provided for your reference:

    """
            examples = 0
            for i,idx in enumerate(indice.cpu().numpy().tolist()):
                if values[i]>thearshold:
                    s = data_set_train[idx]
                    prompt += "Gene: "+s[2]+" Mention: "+s[0]+" Code: "+s[1]+"\n"
                    examples+=1
            prompt+="Now, complete the last entry:\n"+"Gene: "+gene+" Mention: "+text+" Code: "
            model_out = llm_call.run(prompt)
            
            words = model_out.split()
            candidates = [re.sub(r'[^\w\s\|+-]', '', word) for word in words if '|' in word]
            pred = "-"
            if len(candidates) == 1:
                pred = candidates[0]
            elif len(candidates) == 4:
                pred = candidates[-1]
            elif len(candidates) == 0:
                #print('=====')
                #print(prompt)
                #print(model_out)
                #print("error")
                #print('=====')
                pred = "-"
                # break
            else:
                pred = candidates[-1]
            #print('=====')
            #print(model_out)
            #print("pred",pred)
            #print("valid output")
            #print('=====')
            if filter_sub_w_allele(pred):
                _pred = pred
                pred = pred.replace("SUB","Allele")
                
                if pred[-1]=="|":
                    pred=pred[:-1]
                    
                print(f"regex replace, {_pred} to {pred}")
            
            return pred#, prompt, values, examples
        
        return llm_predict
    
    
    
    test_run = load_data(testset)
    

    with open(f'{kb_folder}/NCBI-Gene/gene_lookup.json', 'r') as f:
        gene_lookup = json.load(f)

    print("load training data and kbases for seq_variant")
    training_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_train.json"))
    use_val = True
    if use_val:
        val_data = load_data(os.path.join(dataset_folder, "bc8_biored_task1_val_revealed.json"))
        training_data.extend(val_data)
    
    train_ds, val_ds = load_train_val("all", use_val, kb_folder, dataset_folder)

    data_set_train = set()
    for sample in train_ds:
        
        if "," in sample["nearest_gene"]:
            sample["nearest_gene"] = sample["nearest_gene"].split(",")[0]
        
        if sample["nearest_gene"] in gene_lookup:
            data_set_train.add((convert_amino_acids(sample["text"]), sample["norm"], gene_lookup[sample["nearest_gene"]]))
        else:
            pass
            #print("Gene code not found", sample["nearest_gene"])
    if use_val:
        for sample in val_ds:
            if "," in sample["nearest_gene"]:
                sample["nearest_gene"] = sample["nearest_gene"].split(",")[0]
            
            if sample["nearest_gene"] in gene_lookup:
                data_set_train.add((convert_amino_acids(sample["text"]), sample["norm"], gene_lookup[sample["nearest_gene"]]))
            else:
                pass
                #print("Gene code not found", sample["nearest_gene"])
                
    data_set_train = list(data_set_train)

    batch_size = 64
    embeddings = []
    for batch_text in tqdm(batch_generator(data_set_train, batch_size=batch_size), total=len(data_set_train)//batch_size):
        embeddings.append(get_embeddings(tuple(batch_text)))
    embeddings = torch.concat(embeddings, axis=0) 

    
    llm_predict = build_ollama_predictor(embeddings, data_set_train, 0.6)
    
    #with open(f'{kb_folder}/NCBI-Gene/gene_lookup.json', 'r') as f:
    #    gene_lookup = json.load(f)
        
    with open(f'{kb_folder}/dbSNP/variant_api_results.json', 'r') as f:
        variant_lookup = json.load(f)
    
    train_data_dict = defaultdict(lambda: defaultdict(lambda: set()))
    for doc in training_data:
        gene_ids = [] 
        for entity in doc['entities']:
            if entity['label']=='GeneOrGeneProduct' or entity["label"] == "Gene":
                gene_ids.append((entity['linked_id'], entity['start_span']))


        for entity in doc['entities']:
            if entity['label']=='SequenceVariant':
                distance = 10_000_000 
                for gene in gene_ids:
                    if abs(entity['start_span']-gene[1]) < distance:
                        distance=abs(entity['start_span']-gene[1])
                        nearest_gene=gene[0]

                train_data_dict[nearest_gene][entity['text'].lower()].add(entity['linked_id']) 

    
    
    counter_total = 0
    
    for doc in tqdm(test_run):
        
        #builds (org_id, start) for a doc    
        gene_ids = []
        # var_ids = []
        for entity in doc['entities']:
            if entity['label']=='GeneOrGeneProduct' or entity['label']=='Gene':
                gene_ids.append((entity['linked_id'], entity['start_span']))
        
        for entity in doc['entities']:
        
            if entity['label']=='SequenceVariant':
            
                counter_total+=1

                #looks up nearest org
                distance = 10_000_000 
                for g in gene_ids:
                    if abs(entity['start_span']-g[1]) < distance:  # TODO and org[0] in genes.keys()
                        distance=abs(entity['start_span']-g[1])
                        nearest_g=g[0]
                #handles logic
                #exact match first
                
                prediction = []
                entity["pred_g"] = nearest_g
                # print(entity, nearest_g)
                if len(prediction) == 0:
                    if entity['text'].lower().startswith('rs'):
                        prediction = [entity['text'].lower()]

                if len(prediction) == 0:
                    if nearest_g in gene_lookup:
                        nearest_g = gene_lookup[nearest_g]

                        text = entity['text']
                        text = text.replace('/','>')    
                        text = text.replace('--','')        
                        text = text.replace(' ','')        
                        entity['text'] = text


                        # from training
                        #if len(prediction) == 0:
                        #   if nearest_g in train_data_dict:
                        #       if entity['text'].lower() in dict(train_data_dict[nearest_g]):
                        #           gene_id=dict(train_data_dict[nearest_g])[entity['text'].lower()]
                        #           # gene_id = list(gene_id)[0]
                        #           prediction.extend(list(gene_id))

                        if len(prediction) == 0:
                            # lookup
                            if entity['text']+" "+nearest_g in variant_lookup:
                                rsid = [x["rsid"] for x in variant_lookup[entity['text']+" "+nearest_g] if "rsid" in x]
                            else:
                                rsid = [x["rsid"] for x in get_variant(entity['text'], nearest_g) if "rsid" in x] 
                                variant_lookup[entity['text']+" "+nearest_g] = rsid
                            prediction.extend(rsid)
                            
                if len(prediction) == 0 and llm_call is not None:
                    # try run ollama prediction
                    if nearest_gene in gene_lookup:
                        nearest_gene = gene_lookup[nearest_gene]
                    else:
                        nearest_gene = nearest_gene
                    
                    entity['linked_id'] = [llm_predict(convert_amino_acids(text), nearest_g)]
                else:
                    #print("pred")
                    entity['linked_id'] = prediction
            
    #import pickle
    #with open("_testrun_dbug.p", "wb") as f:
    #    pickle.dump(test_run, f)
                            
    # Desambiguation based on the document
    for doc in test_run:
        id_entities = defaultdict(list)
        for entity in doc["entities"]:
            if entity["label"] == "SequenceVariant":
                if isinstance(entity["linked_id"], list):
                    for linked_id in entity["linked_id"]:
                        id_entities[linked_id].append(entity["id"])
                        # print(id_entities)
                    
        # do majoraty voting (pick the id that has the longest list that each entity belongs too)
        for entity in doc["entities"]:
            if entity["label"] == "SequenceVariant":
                if isinstance(entity["linked_id"], list):
                    candidates = [(linked_id, len(id_entities[linked_id])) for linked_id in entity["linked_id"]]
                    if len(candidates)>0:
                        most_freq_id, _ = max(candidates, key=lambda x:x[1])                    
                        entity["linked_id"] = most_freq_id
                    else:
                        entity["linked_id"] = "-"

    test_run_data_dict = {doc["id"]:{str(ent["id"]):ent for ent in doc["entities"] if ent["label"] == "SequenceVariant"} for doc in test_run}
    
    #import pickle
    #with open("_dbug.p", "wb") as f:
    #    pickle.dump(test_run_data_dict, f)
        
    # build doc
    with open(testset) as f:
        testdata = json.load(f)
    c = 0
    for doc in testdata["documents"]:
        annotated_entities = test_run_data_dict[doc["id"]]
        for passage in doc["passages"]:
            for annotation in passage["annotations"]:
                if str(annotation["id"]) in annotated_entities and annotation["infons"]["type"] == "SequenceVariant":# and annotated_entities[annotation["id"]]["linked_id"] != "-":
                    c +=1
                    annotation["infons"]["identifier"] = annotated_entities[annotation["id"]]["linked_id"]
    
    print("number of predicted seq var:", c)
    with open(output_file,"w") as fOut:
        json.dump(testdata, fOut, indent=2)
        
    return output_file
