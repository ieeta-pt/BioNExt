from src.linker.taxonomy import run_taxonomy
from src.linker.chemicals import run_chemicals
from src.linker.diseases import run_diseases
from src.linker.genes import run_genes
from src.linker.seq_variant import run_seq_variant
from src.linker.cells import run_cells
from src.linker.cleaner import run_cleaner

class Linker:
    
    def __init__(self, llm_api, output_file, dataset_folder, kb_folder) -> None:
        self.llm_api = llm_api
        self.output_file = output_file
        self.dataset_folder = dataset_folder
        self.kb_folder = kb_folder
        self.llm_call = None
        if self.llm_api["module"] is not None:
            module = __import__(self.llm_api["module"])
            class_f = getattr(module, self.llm_api["module"]) 
            del self.llm_api["module"]
            self.llm_call = class_f(**self.llm_api)
    
    def run(self, testset):
        # run linker sequentially
        testset = run_taxonomy(testset, self.output_file, self.dataset_folder, self.kb_folder)
        testset = run_chemicals(testset, self.output_file, self.dataset_folder, self.kb_folder)
        testset = run_diseases(testset, self.output_file, self.dataset_folder, self.kb_folder)
        testset = run_genes(testset, self.output_file, self.dataset_folder, self.kb_folder)
        testset = run_seq_variant(testset, self.output_file, self.dataset_folder, self.kb_folder, self.llm_call)
        testset = run_cells(testset, self.output_file, self.dataset_folder, self.kb_folder)
        testset = run_cleaner(testset, self.output_file)
        return testset
    
    def __str__(self):
        return "Linker"
    
    def __repr__(self) -> str:
        return self.__str__()