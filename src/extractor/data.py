from typing import Any
import torch
import os

import random
import math

# from utils import split_chunks, RangeDict
from collections import defaultdict

from transformers import AutoTokenizer, DataCollatorForTokenClassification

import json
import math

import numpy as np

import itertools

from tqdm import tqdm

def load_data(data, neg_sample_multiplier=1, normalize_order=False):
    #generates the mask for the relation transitions
    mask = []
    for entity1 in  range(6):
        tmp = []
        for entity2 in range(6):
            tmp.append([float(1) for i in range(9)]) 
        mask.append(tmp)

    
    label2idEntity = {"GeneOrGeneProduct": 0, "DiseaseOrPhenotypicFeature": 1, "ChemicalEntity": 2, "SequenceVariant": 3,
            "OrganismTaxon": 4, "CellLine": 5, "Disease":1, "Gene":0, "Chemical":2, "Mutation": 3, "CellLine": 5, "Organism":4 }
    
    label2id = {'Association': 0, 'Positive_Correlation':1, 'Negative_Correlation':2,
               'Cotreatment': 3, 'Bind': 4, 'Comparison': 5, 'Conversion': 6,    
               'Drug_Interaction':7, 'Negative_Class': 8}

    #generate a mask for the possible paris of relations:
    posisble_relation_pairs = []
    for i in range(6):
        tmp = []
        for j in range(6):
            tmp.append(0)
        posisble_relation_pairs.append(tmp)

    #this is bidirectional
    posisble_relation_pairs[label2idEntity['Gene']][label2idEntity['Gene']]=1
    posisble_relation_pairs[label2idEntity['Chemical']][label2idEntity['Disease']]=1
    posisble_relation_pairs[label2idEntity['Disease']][label2idEntity['Gene']]=1
    posisble_relation_pairs[label2idEntity['Chemical']][label2idEntity['Gene']]=1
    posisble_relation_pairs[label2idEntity['Gene']][label2idEntity['Chemical']]=1    
    posisble_relation_pairs[label2idEntity['Disease']][label2idEntity['SequenceVariant']]=1
    posisble_relation_pairs[label2idEntity['SequenceVariant']][label2idEntity['Disease']]=1    
    posisble_relation_pairs[label2idEntity['Chemical']][label2idEntity['Chemical']]=1
    posisble_relation_pairs[label2idEntity['Chemical']][label2idEntity['SequenceVariant']]=1
    posisble_relation_pairs[label2idEntity['SequenceVariant']][label2idEntity['Chemical']]=1 
    posisble_relation_pairs[label2idEntity['SequenceVariant']][label2idEntity['SequenceVariant']]=1 

    posisble_relation_pairs[label2idEntity['Disease']][label2idEntity['Chemical']]=1
    posisble_relation_pairs[label2idEntity['Gene']][label2idEntity['Disease']]=1

    
    clean_data = []
    rels = set()


    
    for i in data:
        entities = i['passages'][0]['annotations']+i['passages'][1]['annotations']
        #list of ids of relations
        if normalize_order:
            ids = sorted(list(set([(e['infons']['identifier'], e['infons']['type']) for e in entities])), key=lambda x:x[1])
        else:
            ids = set([(e['infons']['identifier'], e['infons']['type']) for e in entities])
        rel_ids = set([(r['infons']['entity1'], r['infons']['entity2']) for r in i['relations']])
        
        #extract the last realtion
        if len(i['relations']) != 0:
            last_relation = int(i['relations'][-1]['id'][1:])
        else:
            last_relation = 0
        perms = [i for i in (itertools.combinations(ids, 2))]
        # random.seed(42)
        random.shuffle(perms)
        shuffle_counter = int(len(rel_ids)*neg_sample_multiplier) if len(rel_ids) != 0 else len(perms)
        for e1,e2 in perms:
            if shuffle_counter == 0:
                break
            if posisble_relation_pairs[label2idEntity[e1[1]]][label2idEntity[e2[1]]] == 1:
                if (e1[0],e2[0]) not in rel_ids and (e2[0],e1[0]) not in rel_ids:
                    last_relation+=1
                    shuffle_counter-=1
                    i['relations'].append({"id":f's{last_relation}' ,"infons":{"entity1":e1[0], "entity2": e2[0], "type":'Negative_Class' , "novel":"no"}})
                    

        for r in i['relations']:
            text = i['passages'][0]['text']+' '+i['passages'][1]['text']
            r_ids_entity1 = set()
            r_ids_entity2 = set()
            e_clean = []
            for el in r['infons']['entity1'].split(","):
                r_ids_entity1.add(el)
            for el in r['infons']['entity2'].split(","):
                r_ids_entity2.add(el)
            for e in entities:
                #if r['infons']['type'] != 'Negative_Class' else [e['infons']['identifier']]

                for el in (e['infons']['identifier'].split(",")):
                    
                    if el in r_ids_entity1:
                        e_clean.append({"id":e['id'], "linked_id":e['infons']['identifier'], "label":e['infons']['type'], "start_span":e['locations'][0]['offset'], "end_span":e['locations'][0]['offset']+e['locations'][0]['length'], "text": e['text'], "entity_order":1})
                        label1 = e['infons']['type']
                        break
                    elif el in r_ids_entity2:
                        e_clean.append({"id":e['id'], "linked_id":e['infons']['identifier'], "label":e['infons']['type'], "start_span":e['locations'][0]['offset'], "end_span":e['locations'][0]['offset']+e['locations'][0]['length'], "text": e['text'], "entity_order":2})
                        label2 = e['infons']['type']
                        break

            mask[label2idEntity[label1]][label2idEntity[label2]][label2id[r['infons']['type']]] = float(0)
            mask[label2idEntity[label2]][label2idEntity[label1]][label2id[r['infons']['type']]] = float(0)
            rels.add(tuple([label1, label2]))
            for j in reversed(e_clean):
                
                text = text[:j['end_span']] + f"[e{j['entity_order']}]" + text[j['end_span']:]
                text = text[:j['start_span']] + f"[s{j['entity_order']}]" + text[j['start_span']:]
        

        
            clean_data.append({"id":i['id'], "text":text, "entities": e_clean,
                               "relations":r, 'label': label2id[r['infons']['type']], 'entity1_type':label2idEntity[label1], 'entity2_type':label2idEntity[label2]} )
    return clean_data, mask



def load_train_test_split(
                              folder_path,
                              tokenizer,
                                index_type,
                                special_tokens,
                              test_split_percentage=0.8,
                              train_transformations=None,
                              train_augmentations=None,
                              test_transformations=None,
                                ):

    docs = defaultdict(list)

    #load the file as json
    
    with open(os.path.join(folder_path, "bc8_biored_task1_train.json")) as f:
        data = json.load(f)
    data = data['documents']
    
    
    # random.shuffle(data)
    train_data, mask1 = load_data(data[:int(len(data) * test_split_percentage)])
    test_data, mask2  = load_data(data[int(len(data) * test_split_percentage):])

    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    #this assert fails when the test dataset has a class that the training dataset doesent
    # assert sum((mask2 - mask1).flatten() == -1) == 0

    # if(mask1 != mask2):
    #     print("data leakage")
    mask = mask1 + mask2

    mask[mask == 2] = -9e9
    mask[mask == 1] = float(0)
    
    return  DocumentReaderDatasetForTraining(dataset=train_data,
                                             tokenizer=tokenizer,
                                             special_tokens=special_tokens,
                   
                                             transforms=train_transformations,
                                             augmentations=train_augmentations,
                                            index_type = index_type, 
                                             mask=mask), \
            DocumentReaderDatasetForTraining(dataset=test_data,
                                             tokenizer=tokenizer,
                                             special_tokens=special_tokens,
                        
                                             transforms=test_transformations,
                                            index_type = index_type,
                                            mask=mask)


def load_train(folder_path,
               tokenizer,
               index_type,
               special_tokens,
               neg_sample_multiplier=1,
               train_transformations=None,
               train_augmentations=None):

    docs = defaultdict(list)

    #load the file as json
    
    with open(os.path.join(folder_path, "bc8_biored_task1_train.json")) as f:
        data = json.load(f)
    data = data['documents']
    
    # random.shuffle(data)
    train_data, mask = load_data(data, neg_sample_multiplier)

    mask = np.array(mask)
    
    #this assert fails when the test dataset has a class that the training dataset doesent
    # assert sum((mask2 - mask1).flatten() == -1) == 0

    # if(mask1 != mask2):
    #     print("data leakage")

    mask[mask == 1] = float(-9e9)
    
    return  DocumentReaderDatasetForTraining(dataset=train_data,
                                             tokenizer=tokenizer,
                                             special_tokens=special_tokens,
                   
                                             transforms=train_transformations,
                                             augmentations=train_augmentations,
                                            index_type = index_type, 
                                             mask=mask)
     
#                               folder_path,
#                               tokenizer,
#                                 index_type,
#                                 special_tokens,
#                               train_transformations=None,
#                               train_augmentations=None,
#                               test_transformations=None,
#                                 ):

#     docs = defaultdict(list)

#     #load the file as json
    
#     with open(os.path.join(folder_path, "new_task1_train.json")) as f:
#         train_data = json.load(f)
#     with open(os.path.join(folder_path, "new_task1_val.json")) as f:
#         val_data = json.load(f)
#     train_data = train_data['documents']
#     val_data = val_data['documents']

    
#     train_data, mask1 = load_data(train_data)
#     test_data, mask2  = load_data(val_data)

#     mask1 = np.array(mask1)
#     mask2 = np.array(mask2)
#     #this assert fails when the test dataset has a class that the training dataset doesent
#     print(mask2-mask1)
#     print(f"number of pairs of entity/relations in validation, but not in train {sum((mask2 - mask1).flatten() == -1)}")
#     # assert sum((mask2 - mask1).flatten() == -1) == 0

#     # if(mask1 != mask2):
#     #     print("data leakage")
#     mask = mask1 + mask2

#     mask[mask == 2] = -9e9
#     mask[mask == 1] = float(0)
    
#     return  DocumentReaderDatasetForTraining(dataset=train_data[:100],
#                                              tokenizer=tokenizer,
#                                              special_tokens=special_tokens,
                   
#                                              transforms=train_transformations,
#                                              augmentations=train_augmentations,
#                                             index_type = index_type, 
#                                              mask=mask), \
#             DocumentReaderDatasetForTraining(dataset=test_data,
#                                              tokenizer=tokenizer,
#                                              special_tokens=special_tokens,
                        
#                                              transforms=test_transformations,
#                                             index_type = index_type,
#                                             mask=mask)




def load_inference_data(file_path, folder_path ):

    docs = defaultdict(list)

    #load the file as json
    
    with open(os.path.join(folder_path, file_path)) as f:
        data = json.load(f)
    data = data['documents']
    clean_data = []
    for i in data:
        clean_data.append({"id":i['id'], "text":i['passages'][0]['text']+' '+i['passages'][1]['text'], "abstract_offset": i['passages'][1]['offset']})

    return clean_data

# def load_old_train_test_split(
#                               folder_path,
#                               tokenizer,
#                                 index_type,
#                             special_tokens, 
               
#                               # test_split_percentage=0.15,
#                               train_transformations=None,
#                               train_augmentations=None,
#                               test_transformations=None
#                                 ):
    
#     train_data = load_data("bc8_biored_task1_train.json", folder_path)

#     return  DocumentReaderDatasetForTraining(dataset=train_data[:400],
#                                              tokenizer=tokenizer,
#                                              context_size = context_size,
#                                              transforms=train_transformations,
#                                              augmentations=train_augmentations,
#                                             index_type = index_type), \
#             DocumentReaderDatasetForTraining(dataset=train_data[400:],
#                                              tokenizer=tokenizer,
#                                              context_size=context_size,
#                                              transforms=test_transformations,
#                                             index_type = index_type)

def get_first_and_second(tokenizer):
    first_special_token = None
    second_special_token = None

    valid_token = tokenizer("a", add_special_tokens=False).input_ids

    for token in tokenizer("a", add_special_tokens=True).input_ids:
        if token not in valid_token:
            if first_special_token is None:
                first_special_token = token
            elif second_special_token is None:
                second_special_token = token
            else:
                raise RuntimeError("This model has 3 special tokens?")
            
    return first_special_token, second_special_token

def encode_from_text(tokenizer, doc, index_type, special_tokens, mask=None):

    s1_id, e1_id, s2_id, e2_id = tokenizer("".join(special_tokens), add_special_tokens=False).input_ids
    
    encoding = tokenizer(doc["text"])
            
    tokens = encoding.input_ids
    
    #offsets = encoding.offsets
    attention_mask = encoding.attention_mask

    first_special_token, last_special_token = get_first_and_second(tokenizer)
    
    max_valid_length = tokenizer.model_max_length - int(first_special_token is not None) - int(last_special_token is not None)
    
    tokens = tokens[int(first_special_token is not None):len(tokens)-int(last_special_token is not None)]
    for i in range(math.ceil(len(tokens)/max_valid_length)):
        inputs = tokens[min(i*max_valid_length,(max(0,len(tokens) - max_valid_length))):min((i+1)*max_valid_length, len(tokens))]
        if first_special_token is not None:
            inputs.insert(0, first_special_token)
        if last_special_token is not None:
            inputs.append(last_special_token)
        _inputs = np.array(inputs)
        s_index = ((_inputs == s1_id) | (_inputs == s2_id))
        e_index = ((_inputs == e1_id) | (_inputs == e2_id))
        s_e_indes = s_index | e_index
        
        if doc["relations"]['infons']["entity1"] != doc["relations"]['infons']["entity2"]:
            # verify if the inputs ids have the two entities
            if not ((_inputs == s1_id).any() and (_inputs == s2_id).any()):
                # print("skiped sample")
                # print(doc['text'])
                # print(doc["relations"])
                # print(doc["entities"])
                continue
        
        if index_type == 's':
            indexes = np.where(s_index)
        if index_type == 'e':
            indexes = np.where(e_index)
        if index_type == 'both':
            indexes = np.where(s_e_indes)

        if mask is not None:
            new_mask = mask[doc['entity1_type']][doc['entity2_type']]
        else:
            new_mask = None
        
        yield {
            "input_ids":inputs,
            "attention_mask": attention_mask[min(i*tokenizer.model_max_length,(max(0,len(attention_mask) - tokenizer.model_max_length))):min((i+1)*tokenizer.model_max_length, len(attention_mask))],
            "mask": new_mask,
            "indexes": indexes[0].tolist(),
            "sequence_id": i,
        }
class DocumentReaderDatasetForTraining(torch.utils.data.Dataset):

    def __init__(self,
                 dataset,
                 tokenizer,
                 special_tokens,
                 transforms=None,
                 augmentations=None,
                index_type='both',
                mask=None):
        super().__init__()

        self.dataset = []
        self.transforms = transforms
        self.augmentations = augmentations
        
        
        total_collisions=0
        total_new_collisions = 0
        # read txt
        counter = 0
        for doc in dataset:
            for encoded_data in encode_from_text(tokenizer, doc, index_type, special_tokens, mask=mask):

                sample = {
                    "text": doc["text"],
                    "doc_id": doc["id"],
                    "rel_id": doc["relations"]["id"],
                    "label": doc['label'],
                    "novel": int(doc['relations']['infons']['novel'] == "Novel"),
                    "entities": doc["entities"],
                    "relations": doc["relations"],
                    **encoded_data
                }          
                assert len(sample["input_ids"])<=tokenizer.model_max_length
                #assert len(sample["offsets"])<=tokenizer.model_max_length

                    #"annotations": sample_annotations,
                    #"list_annotations": sample_annotations.get_all_annotations(),
                                    #"offsets": offsets[min(i*512,(max(0,len(offsets) - 512))):min((i+1)*512, len(offsets))],

                
                if self.transforms:
                    for transform in self.transforms:
                        sample = transform(sample)
    
                self.dataset.append(sample)
            

        self.tokenizer = tokenizer

        if total_collisions>0:
            print(f"Warning, we found {total_collisions} collisions that were automaticly handle by merging strategy")

        if total_new_collisions>0:
            print(f"WARNING!!! total new collision is {total_new_collisions}, this should be 0")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if self.augmentations:
            for augmentation in self.augmentations:
                sample = augmentation(sample)

        return sample

class SelectModelInputs:

    def __init__(self, inputs_keys):
        self.inputs_keys = inputs_keys
    
    def __call__(self, sample) -> Any:
        return { k:sample[k] for k in self.inputs_keys}
        
def label2int(label):
    #GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'SequenceVariant', 'OrganismTaxon', 'CellLine'
    if label == 'GeneOrGeneProduct':
        return 0
    elif label == 'DiseaseOrPhenotypicFeature' or label == 'Disease':
        return 1
    elif label == 'ChemicalEntity':
        return 2
    elif label == 'SequenceVariant':
        return 3
    elif label == 'OrganismTaxon':
        return 4
    elif label == 'CellLine':
        return 5
    else:
        print("something is wrong")
        print(label)
        return

        
class BIOTagger():
    
    def __call__(self, sample) -> Any:
        
        labels = [0]
        prev_annotation = None
        current_label = 0
        
        for offset in sample["offsets"][1:]:
            if offset is None:
                current_label = 0
            else:
                if offset in sample["annotations"]:

                    if prev_annotation != sample["annotations"][offset]:
                        current_label = 2*label2int(sample["annotations"][offset]["label"])+1
                        prev_annotation = sample["annotations"][offset]
                    else:
                        current_label = 2*label2int(sample["annotations"][offset]["label"])+2
                else:
                    current_label = 0
                    prev_annotation = None
                
            labels.append(current_label)
                    
        #pad
        labels = labels + [0]*(len(sample["offsets"])-len(labels))
        
        #assert len(labels)==len(sample["offsets"])
        #assert len(labels)<=512
        
        return sample | {"labels": labels}

#if prev_annotation is None:
#    current_label = 1
#    prev_annotation = sample["annotations"][offset]
#else:      

# O O O B I I O O O
# entity identification

# O O O O B O O

class RandomlyUKNTokens:

    def __init__(self, 
                 tokenizer, 
                 context_size,
                 prob_change=0.5, 
                 percentage_changed_tags=0.2):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.prob_change = prob_change
        self.percentage_changed_tags = percentage_changed_tags
    
    def pick_token(self):
        return self.tokenizer.unk_token_id
    
    def __call__(self, sample) -> Any:
        
        if torch.rand(1) < self.prob_change:
            
            # pick tokens based on the tags, same amount of O and B/I
            
            # get total of BI and O tags
            bi_tags = []
            o_tags = []
            
            # when the sample ends it may not have right context
            right_context = max(self.context_size - (self.tokenizer.model_max_length - len(sample["labels"])), 0)
            
            if right_context == 0:
                labels = sample["labels"][self.context_size:]
            else:
                labels = sample["labels"][self.context_size:-right_context]

            for i,tag in enumerate(labels):
                if tag==0:
                    o_tags.append(i+self.context_size)
                else:
                    bi_tags.append(i+self.context_size)
            
            num_changes = int(self.percentage_changed_tags*len(bi_tags))
            if num_changes==0 and len(bi_tags)>0:
                num_changes=1
            
            bi_rand_indexes = torch.randperm(len(bi_tags))[:num_changes]
            o_rand_indexes = torch.randperm(len(o_tags))[:num_changes]
            
            for i in bi_rand_indexes:
                sample["input_ids"][i] = self.pick_token()
                
            for i in o_rand_indexes:
                sample["input_ids"][i] = self.pick_token()
            
        return sample
    
class RandomlyReplaceTokens(RandomlyUKNTokens):
    def pick_token(self):
        token_id = int(torch.rand(1)*self.tokenizer.vocab_size)

        while token_id in [self.tokenizer.unk_token_id,self.tokenizer.pad_token_id,self.tokenizer.sep_token_id,self.tokenizer.cls_token_id]:
            token_id = int(torch.rand(1)*self.tokenizer.vocab_size)
            
        return token_id
    
    
class EvaluationDataCollator(DataCollatorForTokenClassification):
    
    def torch_call(self, features):
        
        model_inputs = {"input_ids", "attention_mask"}
        
        reminder_columns = set(features[0].keys()) - model_inputs
        
        out = {k:[] for k in reminder_columns}
        inputs = [{k: feature[k] for k in model_inputs}
                  for feature in features ]
        
        for feature in features:
            for k in reminder_columns:
                out[k].append(feature[k])
        
        out["inputs"] = super().torch_call(inputs)
        
        return out
        
        
        

class TrainingNegConfidenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 sample_file, 
                 sampler_class, 
                 tokenizer,
                 special_tokens,
                 transforms=None,
                 augmentations=None,
                 index_type='both',
                 mask=None,
                 neg_multiplier=1):
        super().__init__()
        self.pos_doc_list = defaultdict(list)
        self.neg_doc_list = defaultdict(list)
        self.neg_multiplier = neg_multiplier
        self.sampler_class = sampler_class
        self.epoch = -1

        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        if transforms is None:
            # maybe slow? check how to improve identity lambda function
            self.transformations = lambda x:x
        else:
            def _transforms(x):
                for transform in transforms:
                    x = transform(x)
                return x
            self.transformations = _transforms
        self.transforms = transforms
        self.index_type = index_type
        self.augmentations = augmentations
        self.mask = mask

                    
        
        with open(sample_file) as f:
            for doc in tqdm(map(json.loads, f)):
                for encoded_data in encode_from_text(tokenizer, doc, index_type, special_tokens, mask=mask):

                    sample = {**doc,**encoded_data}  
                            
                    if "negative_class_confidence" in sample:
                        self.neg_doc_list[sample["doc_id"]].append(sample)
                    else:
                        self.pos_doc_list[sample["doc_id"]].append(sample)
                    
        self.len_pos = {k:len(v) for k, v in self.pos_doc_list.items()}
        self.len_neg = {k:len(v) for k, v in self.neg_doc_list.items()}
        self.pos_doc_list = [(k, sample) for k, v in self.pos_doc_list.items() for sample in v]
    
    def __len__(self):
        number_pos_docs = sum([x for x in self.len_pos.values()]) #sum([len(docs) for docs in self.pos_doc_list.values()])
        number_neg_docs = min(sum([x for x in self.len_neg.values()]), number_pos_docs*self.neg_multiplier ) #sum([len(docs) for docs in self.pos_doc_list.values()])
        number_docs = number_pos_docs+number_neg_docs
        # number_docs += number_pos_docs*self.neg_multiplier
        return number_docs
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert  worker_info is None or worker_info.num_workers<=1, "Split by workers not implemented"
        self.epoch += 1
        sampler = self.sampler_class(self.epoch)
        
        return NegativeConfidenceIterator(pos_samples=self.pos_doc_list,
                                          len_pos=self.len_pos,
                                          neg_samples=self.neg_doc_list,
                                          sampler=sampler,
                                          epoch=self.epoch,
                                          transformations=self.transformations,
                                          neg_multiplier=self.neg_multiplier,
                                          
                                          )

class HighConfidenceSampler():
    def __init__(self, epoch):
        self.epoch=epoch

    def __call__(self, neg_samples, len_pos, neg_multiplier):

        sorted_neg_docs = {}
        for doc_id, neg_samples in neg_samples.items():

            # fix this
            _temp = sorted(neg_samples, key=lambda x:x["negative_class_confidence"])
            
            if doc_id in len_pos:
                sorted_neg_docs[doc_id] = _temp[:int(len_pos[doc_id]*neg_multiplier)]
            else:
                sorted_neg_docs[doc_id] = _temp
        
        return sorted_neg_docs
        
def probabilistic_round(x):
    return int(math.floor(x + random.random()))

class NegativeConfidenceIterator():
    def __init__(self, pos_samples, len_pos, neg_samples, sampler, epoch, transformations, neg_multiplier=1):
        self.pos_samples = pos_samples
        self.len_pos = len_pos
        self.neg_samples = neg_samples
        self.sampler = sampler
        self.epoch = epoch
        self.transformations=transformations
        self.neg_multiplier = neg_multiplier

        negatives_without_positives = set(neg_samples.keys()) - set(len_pos.keys())
        
        def sampler_loop():
            random.shuffle(self.pos_samples)
            negs = self.sampler(self.neg_samples, self.len_pos, self.neg_multiplier)

            
            for doc_id, pos_sample in self.pos_samples:
                yield self.transformations(pos_sample)

                # select a neg sample
                if doc_id in negs:
                    for neg_sample in random.choices(negs[doc_id], k=probabilistic_round(self.neg_multiplier)):
                        yield self.transformations(neg_sample)

            # do not like this part
            for doc_id in negatives_without_positives:
                for neg_sample in negs[doc_id]:
                    yield self.transformations(neg_sample)
           
        self.iterator = iter(sampler_loop())
        
    def __next__(self):
        return next(self.iterator)

