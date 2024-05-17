import os
import yaml
from transformers import TrainingArguments, AutoTokenizer, BertConfig

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open



def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

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
                l.append(self.data[i]) # list with all the annotaiton that colide
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


