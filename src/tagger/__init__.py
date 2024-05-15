from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForMaskedLM,AutoModel


from src.tagger.decoder import decoder


from src.data import load_train_test_split, BIOTagger, SelectModelInputs, EvaluationDataCollator, load_inference_data, ID2LABEL_ENTITY

from torch.utils.data import DataLoader
import torch
import pandas as pd
from collections import defaultdict 
from tqdm import tqdm
import os
import json

def decoder_from_samples(prediction_batch, context_size):
    
    documents = {}
    padding = context_size

    # reconsturct the document in the correct order
    for i in range(len(prediction_batch)):
        doc_id = prediction_batch[i]['doc_id']
        if doc_id not in documents.keys():
            documents[doc_id] = {}
            
            # run 1 time is enough for this stuff

        documents[doc_id][prediction_batch[i]['sequence_id']] = {
            'output': prediction_batch[i]['output'],
            'offsets': prediction_batch[i]['offsets'],
            'text':prediction_batch[i]["text"]}

    print("DOCUMENTS:", len(documents))

    predicted_entities = {}
    # decode each set of labels and store the offsets
    for doc in documents.keys():
        text = documents[doc][0]["text"]
        current_doc = [documents[doc][seq]['output'] for seq in sorted(documents[doc].keys())]
        current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
        predicted_entities[doc] = {"decoder": decoder(current_doc, current_offsets, padding=padding, text=text), "document_text": text}
    return predicted_entities

def remove_txt(data):
    new_data = {}
    for k,v in data.items():
        new_k, _ = os.path.splitext(k)
        new_data[new_k]=v
        
    return new_data

class Tagger:
    def __init__(self, checkpoint, trained_model_path, output_folder, batch_size) -> None:
        
        # load model
        device="cpu"
        if torch.cuda.is_available():
            device="cuda"
            #single GPU bc CRF
            assert torch.cuda.device_count()==1, "CRF is not prepared to run as SPMD."

        self.output_folder = output_folder
        self.batch_size = batch_size
        self.device = device
        self.model = AutoModel.from_pretrained(checkpoint, 
                                               trust_remote_code=True,
                                               cache_dir=trained_model_path)
        self.config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                                       cache_dir=trained_model_path)
        #self.model, self.tokenizer, self.config = load_model_and_tokenizer_locally(checkpoint, trained_model_path)
        self.model = self.model.to(device)

        self.tokenizer.model_max_length = 512

    def __str__(self):
        return "Tagger"
    
    def __repr__(self) -> str:
        return self.__str__()
        
    def run(self, testset):
        
        test_ds = load_inference_data(testset, tokenizer=self.tokenizer, context_size=self.config.context_size)

        eval_datacollator = EvaluationDataCollator(tokenizer=self.tokenizer, 
                                                padding=True,
                                                label_pad_token_id=self.tokenizer.pad_token_id)
    
        dl = DataLoader(test_ds, batch_size=self.batch_size, collate_fn=eval_datacollator)
    
        outputs = []
        for train_batch in tqdm(dl):
            with torch.no_grad():
                train_batch["output"] = self.model(**train_batch["inputs"].to(self.device)).type(torch.int32).numpy().tolist()
                train_batch |= train_batch["inputs"]
                del train_batch["inputs"]
            keys = list(train_batch.keys()) + ["output"]
            for i in range(len(train_batch["output"])):
                outputs.append({k:train_batch[k][i] for k in keys})
                
        predicted_entities = decoder_from_samples(outputs, context_size=self.config.context_size)
        #predicted_entities = remove_txt(predicted_entities)
        
        # build doc
        with open(testset) as f:
            testdata = json.load(f)
        
        for doc in testdata["documents"]:
            annotations = predicted_entities[doc["id"]]
            title_len = doc["passages"][1]["offset"]

            for i,span in enumerate(annotations["decoder"]["span"]):

                if span[0] < title_len:
                    current_passage_index=0
                else:
                    current_passage_index=1
                
                doc["passages"][current_passage_index]["annotations"].append({
                    "id": str(i),
                    "infons": {
                        "type": ID2LABEL_ENTITY[span[-1]],
                        "identifier": "-"
                    },
                    "text": annotations["decoder"]["text"][i],
                    "locations": [
                    {
                    "offset": span[0],
                    "length": span[1]-span[0]
                    }
                ]
                })
                # annotations in title
                
                # annotations in abstract
        
        output_filename = os.path.join(self.output_folder, os.path.basename(testset))
        with open(output_filename, "w") as fOut:
            json.dump(testdata, fOut, indent=2)

        return output_filename