from src.linker.taxonomy import run_taxonomy
from src.linker.chemicals import run_chemicals
from src.linker.diseases import run_diseases
from src.linker.genes import run_genes
from src.linker.seq_variant import run_seq_variant
from src.linker.cells import run_cells
from src.linker.cleaner import run_cleaner
from src.linker.utils import maybe_download_dataset, maybe_download_kb
import os 

class Linker:
    
    def __init__(self, llm_api, output_folder, dataset_folder, kb_folder) -> None:
        self.llm_api = llm_api
        self.output_folder = output_folder
        self.dataset_folder = dataset_folder
        self.kb_folder = kb_folder
        self.llm_call = None
        if self.llm_api["module"] is not None:
            module = __import__(self.llm_api["module"])
            class_f = getattr(module, self.llm_api["module"]) 
            del self.llm_api["module"]
            self.llm_call = class_f(**self.llm_api)
        
        maybe_download_dataset(self.dataset_folder)
        maybe_download_kb(self.kb_folder)
        
    
    def run(self, testset):
        # run linker sequentially
        output_filename = os.path.join(self.output_folder, os.path.basename(testset))
        
        testset = run_taxonomy(testset, output_filename, self.dataset_folder, self.kb_folder)
        testset = run_chemicals(testset, output_filename, self.dataset_folder, self.kb_folder)
        testset = run_diseases(testset, output_filename, self.dataset_folder, self.kb_folder)
        testset = run_genes(testset, output_filename, self.dataset_folder, self.kb_folder)
        testset = run_seq_variant(testset, output_filename, self.dataset_folder, self.kb_folder, self.llm_call)
        testset = run_cells(testset, output_filename, self.dataset_folder, self.kb_folder)
        testset = run_cleaner(testset, output_filename)
        return output_filename
    
    def __str__(self):
        return "Linker"
    
    def __repr__(self) -> str:
        return self.__str__()