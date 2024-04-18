import click
import json
import os

def run_cleaner(testset, output_file):
    total_annotation = 0
    total_valid_annotation = 0
    # remove entities that do not have been normalized
    with open(testset) as f:
        testdata = json.load(f)
    c = 0
    for doc in testdata["documents"]:
        atomic_annotation_counter = 0
        for passage in doc["passages"]:
            clean_annotations = []
            for annotation in passage["annotations"]:
                total_annotation += 1
                if annotation["infons"]["identifier"] != "-":
                    annotation["id"] = atomic_annotation_counter
                    clean_annotations.append(annotation)
                    atomic_annotation_counter+=1
                    total_valid_annotation+=1
                    
            passage["annotations"] = clean_annotations
            
    print("number of total ann:", total_annotation, "after clean:", total_valid_annotation)
    with open(output_file,"w") as fOut:
        json.dump(testdata, fOut, indent=2)
    
    return output_file