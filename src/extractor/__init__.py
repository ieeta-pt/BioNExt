import json
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModel, AutoTokenizer
from src.extractor.data import load_data, DocumentReaderDatasetForTraining
from src.extractor.collator import DataCollatorForInference
import os

def load_data_for_inference(file_path,
                              tokenizer,
                                index_type,
                                special_tokens,
                          transformations=None,
                          augmentations=None,
                                ):

    #load the file as json
    
    with open(file_path) as f:
        data = json.load(f)
    data = data['documents']

    data, _  = load_data(data)

    print("load_data_for_inference", len(data))
    
    return  DocumentReaderDatasetForTraining(dataset=data,
                                             tokenizer=tokenizer,
                                             special_tokens=special_tokens,
                                             transforms=transformations,
                                             augmentations=augmentations,
                                            index_type = index_type, 
                                             mask=None)

class Extractor:
    
    def __init__(self, checkpoint, trained_model_path, output_folder, batch_size) -> None:
        self.output_folder = output_folder
        self.batch_size = batch_size
        
        #self.model, self.tokenizer, self.config = load_model_local(checkpoint, "cuda")
        self.model = AutoModel.from_pretrained(checkpoint, 
                                               trust_remote_code=True,
                                               cache_dir=trained_model_path)
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                                       cache_dir=trained_model_path)
        
        self.model = self.model.to("cuda")
        # comment this, when models have this specified
        #self.config.tokenizer_special_tokens = ['[s1]','[e1]', '[s2]','[e2]' ] 
         


    def run(self, testset):
        
        dataset = load_data_for_inference(testset, self.tokenizer, self.config.index_type, self.config.tokenizer_special_tokens)
        print(len(dataset), self.batch_size)
        
        colate = DataCollatorForInference(tokenizer=self.tokenizer, padding="longest", inputs_keys=["input_ids", "attention_mask", "indexes"])

        dl = torch.utils.data.DataLoader(dataset, collate_fn=colate, batch_size=self.batch_size)

        doc_relations = defaultdict(list)
        
        with torch.no_grad():
            for batched_sample, metadata in tqdm(dl, total = len(dataset)//self.batch_size):
                relation_logits, novel_logits = self.model(**batched_sample.to("cuda")).logits
                relation_logits = relation_logits.cpu()
                novel_logits = novel_logits.cpu()
                
                relation_class = np.argmax(relation_logits, axis=-1)
                novel_class = np.argmax(novel_logits, axis=-1)
                
                for i in range(relation_class.shape[0]):
                    if relation_class[i] != 8:
                        entity1 = metadata["relations"][i]["infons"]["entity1"]
                        entity2 = metadata["relations"][i]["infons"]["entity2"]
                        doc_relations[metadata["doc_id"][i]].append({"entity1":entity1,"entity2":entity2,"label": int(relation_class[i]), "novel": int(novel_class[i]), "sequence_id":  metadata["sequence_id"][i], "relation_logits": relation_logits[i].numpy().tolist(), "novel_class": novel_logits[i].numpy().tolist()})
        
        # Prepare and write!
        
        remapped_data = defaultdict(lambda: defaultdict(list))
        for k,v in doc_relations.items():
            for pair in v:
                entity_1 = pair["entity1"].split(",")
                entity_2 = pair["entity2"].split(",")
                for entity1_part in entity_1:
                    for entity2_part in entity_2:
                        remapped_data[k][(entity1_part,entity2_part)].append([pair['label'],pair['novel'],pair['sequence_id'],torch.softmax(torch.Tensor(pair['relation_logits'])[0], dim =-1).tolist(),pair['novel_class'][0]])
        #remapped data is a dict[doc][(e1,e2)][list(label, novel ...)]
        
        
        for doc, en in remapped_data.items():
            
            for k,v in en.items():
                #check if multpiple mapping exist
                if len(v) > 1:
                    #if it does do math
                    rel_sum = np.sum(np.array([i[3] for i in v]), axis = 0)
                    true_label = np.argmax(rel_sum)
                    novel_sum = np.sum(np.array([i[4] for i in v]), axis = 0)
                    novel_label = np.argmax(novel_sum)
                    remapped_data[doc][k] = [[true_label,novel_label, 0 ,rel_sum, novel_sum]]

        label2id = {'Association': 0, 'Positive_Correlation':1, 'Negative_Correlation':2,
                'Cotreatment': 3, 'Bind': 4, 'Comparison': 5, 'Conversion': 6,    
                'Drug_Interaction':7, 'Negative_Class': 8}

        id2label = {v:k for k,v in label2id.items()}
        id2labelNovel = {1:'Novel', 0:'No'}

        with open(testset) as f:
            data = json.load(f)
            
        data['documents'] = data['documents']
        for i in range(len(data['documents'])):
            id = data['documents'][i]['id']
            rel = []
            counter = 0
        
            for k,v in remapped_data[id].items():
                if len(v) != 0 and v[0][0]!=8:
                    rel.append({'id': f"R{counter}", 'infons':{'entity1': k[0],'entity2':k[1],'type':id2label[v[0][0]],'novel':id2labelNovel[v[0][1]], }})
                    counter += 1
                else: 
                    print("Doc:",id,"does not contains relations.") 
                
            data['documents'][i]['relations'] = rel
        #output
        output_filename = os.path.join(self.output_folder, os.path.basename(testset))

        print("writing to", output_filename)
        with open(output_filename, "w") as fOut:
            json.dump(data, fOut, indent=2)
        
        return output_filename
    
    def __str__(self):
        return "Extractor"
    
    def __repr__(self) -> str:
        return self.__str__()
