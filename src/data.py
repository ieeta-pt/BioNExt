from typing import Any
import torch
import os
import pandas as pd
import random
import math

from src.utils import split_chunks, RangeDict
from collections import defaultdict

from transformers import AutoTokenizer, DataCollatorForTokenClassification

import json


LABEL2ID_ENTITY = {"GeneOrGeneProduct": 0, "DiseaseOrPhenotypicFeature": 1, "ChemicalEntity": 2, "SequenceVariant": 3,
            "OrganismTaxon": 4, "CellLine": 5}#, "Disease":1, "Gene":0, "Chemical":2, "Mutation": 3, "CellLine": 5, "Organism":4 

ID2LABEL_ENTITY = { v:k for k,v in LABEL2ID_ENTITY.items()}


def load_data(file_path):
    #load the file as json
    
    with open(file_path) as f:
        data = json.load(f)
    data = data['documents']
    clean_data = []
    for i in data:
        entities = i['passages'][0]['annotations']+i['passages'][1]['annotations']
        e_clean = []
        for e in entities:
            e_clean.append({"id":e['id'], "linked_id":e['infons']['identifier'], "label":e['infons']['type'], "text":e['text'], "start_span":e['locations'][0]['offset'], "end_span":e['locations'][0]['offset']+e['locations'][0]['length']})
        clean_data.append({"id":i['id'], "text":i['passages'][0]['text']+' '+i['passages'][1]['text'], "entities": e_clean,
                           "relations":i['passages'][0]['relations'] + i['passages'][1]['relations']})

    return clean_data

def load_inference_data(file_path,
                        tokenizer,
                        context_size = 64):

    test_data = load_data(file_path)

    return  DocumentReaderDatasetForTraining(dataset=test_data,
                                             tokenizer=tokenizer,
                                             context_size = context_size,
                                             transforms=None,
                                             augmentations=None)

def load_train_test_split(
                              folder_path,
                              tokenizer,
                              context_size=64,
                              # test_split_percentage=0.15,
                              train_transformations=None,
                              train_augmentations=None,
                              test_transformations=None):

    train_data = load_data(os.path.join(folder_path, "bc8_biored_task1_train.json"))
    test_data = load_data(os.path.join(folder_path,"bc8_biored_task1_val.json"))

    return  DocumentReaderDatasetForTraining(dataset=train_data,
                                             tokenizer=tokenizer,
                                             context_size = context_size,
                                             transforms=train_transformations,
                                             augmentations=train_augmentations), \
            DocumentReaderDatasetForTraining(dataset=test_data,
                                             tokenizer=tokenizer,
                                             context_size=context_size,
                                             transforms=test_transformations)


def load_full_train(folder_path,
                    tokenizer,
                    context_size=64,
                    # test_split_percentage=0.15,
                    transformations=None,
                    augmentations=None):
    
    train_data = load_data(os.path.join(folder_path, "bc8_biored_task1_train.json"))
    test_data = load_data(os.path.join(folder_path,"bc8_biored_task1_val_revealed.json"))

    train_ds =  DocumentReaderDatasetForTraining(dataset=train_data,
                                             tokenizer=tokenizer,
                                             context_size = context_size,
                                             transforms=transformations,
                                             augmentations=augmentations)
    val_ds = DocumentReaderDatasetForTraining(dataset=test_data,
                                             tokenizer=tokenizer,
                                             context_size=context_size,
                                             transforms=transformations,
                                             augmentations=augmentations)
    
    train_ds.merge(val_ds)
    return train_ds

def load_old_train_test_split(
                              folder_path,
                              tokenizer,
                              context_size=64,
                              # test_split_percentage=0.15,
                              train_transformations=None,
                              train_augmentations=None,
                              test_transformations=None):
    
    train_data = load_data(os.path.join(folder_path, "bc8_biored_task1_train.json"))

    return  DocumentReaderDatasetForTraining(dataset=train_data[:400],
                                             tokenizer=tokenizer,
                                             context_size = context_size,
                                             transforms=train_transformations,
                                             augmentations=train_augmentations), \
            DocumentReaderDatasetForTraining(dataset=train_data[400:],
                                             tokenizer=tokenizer,
                                             context_size=context_size,
                                             transforms=test_transformations)

class DocumentReaderDatasetForTraining(torch.utils.data.Dataset):

    def __init__(self,
                 dataset,
                 tokenizer,
                 context_size=64,
                 transforms=None,
                 augmentations=None):
        super().__init__()

        self.context_size = context_size -1 # cls + sep
        self.center_tokens = tokenizer.model_max_length - 2*context_size
        self.dataset = []
        self.transforms = transforms
        self.augmentations = augmentations

        total_collisions=0
        total_new_collisions = 0
        # read txt
        for doc in dataset:
            # get annotations and resolve conflicting ones
            # resolve annotations conflit here?
            sample_annotations = RangeDict()

            new_annotation_index = 0
            #naive and ignore titles
            for annotation in doc['entities']:
                new_span = sample_annotations.maybe_merge_annotations(annotation)

                if new_span:
                    new_annotation_index += 1

                    # lets create a new annotation bc collision
                    annotation = {
                        "ann_id": f"NT{new_annotation_index}",
                        #"label": "PROCEDIMENTO",
                        "start_span": new_span[0],
                        "end_span": new_span[1],
                        "text": doc["text"][new_span[0]:new_span[1]],
                    }

                    #doc_id = doc["filename"]
                    #t = annotation["ann_id"]
                    total_collisions+=1
                    #print(f"File: {doc_id} has collision, new annotation was created {t} span {(annotation['start_span'], annotation['end_span'])}")

                sample_annotations[(annotation["start_span"], annotation["end_span"])] = annotation

            doc["annotations"] = sample_annotations.get_all_annotations()

            encoding = tokenizer(doc["text"], add_special_tokens=False)[0]
            tokens = encoding.ids
            offsets = encoding.offsets

            # add pad tokens to the beggining
            attention_mask = [0] * self.context_size + [1] * len(tokens)
            tokens = [tokenizer.pad_token_id] * self.context_size + tokens
            offsets = [None] * self.context_size + offsets


            #assert len(tokens)==len(offsets)

            for j,i in enumerate(range(self.context_size,len(tokens),self.center_tokens)):

                left_context_tokens = tokens[i-self.context_size:i]
                central_tokens = tokens[i:i+self.center_tokens]
                right_context_tokens = tokens[i+self.center_tokens:i+self.center_tokens+self.context_size]

                left_context_offsets = offsets[i-self.context_size:i]
                central_offsets = offsets[i:i+self.center_tokens]
                right_context_offsets = offsets[i+self.center_tokens:i+self.center_tokens+self.context_size]

                left_context_attention_mask = attention_mask[i-self.context_size:i]
                central_attention_mask = attention_mask[i:i+self.center_tokens]
                right_context_attention_mask = attention_mask[i+self.center_tokens:i+self.center_tokens+self.context_size]

                sample_tokens = [tokenizer.cls_token_id] + left_context_tokens + central_tokens + right_context_tokens + [tokenizer.sep_token_id]
                sample_offsets = [None] + left_context_offsets + central_offsets + right_context_offsets + [None]
                sample_attention_mask = [1] + left_context_attention_mask + central_attention_mask + right_context_attention_mask + [1]

                assert len(sample_tokens)<=tokenizer.model_max_length and len(sample_offsets)<=tokenizer.model_max_length

                if j==0:
                    low_offset, high_offset = sample_offsets[self.context_size+1][0], sample_offsets[-2][1]
                else:
                    low_offset, high_offset = sample_offsets[1][0], sample_offsets[-2][1]

                sample_annotations = RangeDict()

                new_annotation_index = 0
                total_new_collisions = 0

                for annotation in doc["entities"]:
                    if (annotation["start_span"] >= low_offset and annotation["start_span"]<=high_offset or
                            annotation["end_span"] >= low_offset and annotation["end_span"]<=high_offset):

                        new_span = sample_annotations.maybe_merge_annotations(annotation)

                        if new_span:
                            new_annotation_index += 1

                            # lets create a new annotation bc collision
                            annotation = {
                                "ann_id": f"NT{new_annotation_index}",
                                "label": "PROCEDIMENTO",
                                "start_span": new_span[0],
                                "end_span": new_span[1],
                                "text": doc["text"][new_span[0]:new_span[1]],
                            }

                            #doc_id = doc["filename"]
                            #t = annotation["ann_id"]
                            total_new_collisions+=1
                            #print(f"File: {doc_id} has collision, new annotation was created {t} span {(annotation['start_span'], annotation['end_span'])}")

                        sample_annotations[(annotation["start_span"], annotation["end_span"])] = annotation

                #sample_annotations_list = sorted([ annotation for annotation in doc["annotations"] if annotation["start_span"] >= low_offset and annotation["start_span"]<=high_offset or annotation["end_span"] >= low_offset and annotation["end_span"]<=high_offset], key=lambda x:x["start_span"])

                sample = {
                    "text": doc["text"],
                    "doc_id": doc["id"],
                    "sequence_id": j,
                    "input_ids": sample_tokens,
                    "attention_mask": sample_attention_mask,
                    "offsets": sample_offsets,
                    "view_offset": (low_offset, high_offset),
                    "annotations": sample_annotations,
                    "list_annotations": sample_annotations.get_all_annotations(),
                    "og_annotations": doc["annotations"],
                }

                assert len(sample["input_ids"])<=tokenizer.model_max_length
                assert len(sample["offsets"])<=tokenizer.model_max_length

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
    
    def merge(self, dataset):
        # same set of transformations
        assert self.transforms == dataset.transforms
        self.dataset.extend(dataset.dataset)


class SelectModelInputs():
    
    def __call__(self, sample) -> Any:
        return { k:sample[k] for k in ["input_ids", "attention_mask", "labels"]}

        #assert len(samples["input_ids"])<=512
        #assert len(samples["attention_mask"])<=512
        #assert len(samples["labels"])<=512
        
        #return samples
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
                sample["input_ids"][bi_tags[i]] = self.pick_token()
                
            for i in o_rand_indexes:
                sample["input_ids"][o_tags[i]] = self.pick_token()
            
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
        
        