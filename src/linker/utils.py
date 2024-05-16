import os
import requests
from tqdm import tqdm
import re
import zipfile
import json

NORMALIZER_MODEL_MAPPINGS = {
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large": "sapBERT-multilanguage-large",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "sapBERT-multilanguage",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "sapBERT-english",
}

NORMALIZER_MODEL_MAPPINGS_REVERSED = {v:k for k,v in NORMALIZER_MODEL_MAPPINGS.items()}


def download_and_unzip_file(url, local_filename):
    
    download_file(url, local_filename)
    
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(os.path.dirname(local_filename))
        
    os.remove(local_filename)

def download_file(url, local_filename=None):
    """
    Downloads a file from a given URL and saves it locally with a progress bar.

    :param url: URL to the file
    :param local_filename: local path to save the file; if None, uses the last segment of the URL
    :return: None
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]

    # Get the response from the URL
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for request errors
        # Get the total file size from headers
        total_size = int(r.headers.get('content-length', 0))
        
        # Use tqdm to show progress bar
        with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    print(f"File downloaded: {local_filename}")

def load_json_dataset(filepath):
    with open(filepath, mode='r', encoding='utf-8') as fp:
        data = json.loads(fp.read())
    return data



def convert_train_val_files(bc8_biored_train_data_fp, 
                            bc8_biored_val_data_fp, 
                            biored_train_data_fp,
                            biored_dev_data_fp,
                            biored_test_data_fp):
    
    bc8_biored_train_data = load_json_dataset(bc8_biored_train_data_fp)
    bc8_biored_val_data = load_json_dataset(bc8_biored_val_data_fp)

    bc8_biored_subsets = (bc8_biored_train_data, bc8_biored_val_data)
    bc8_biored_subsets_names = ('BC8_BioRED_train', 'BC8_BioRED_val')
    
    biored_train_data = load_json_dataset(biored_train_data_fp)
    biored_dev_data = load_json_dataset(biored_dev_data_fp)
    biored_test_data = load_json_dataset(biored_test_data_fp)

    biored_subsets = (biored_train_data, biored_dev_data, biored_test_data)
    biored_subsets_names = ('BioRED_Train.BioC.JSON', 'BioRED_Dev.BioC.JSON', 'BioRED_Test.BioC.JSON')
    
    bc8_biored_val_revealed_data_fp = os.path.join(os.path.dirname(bc8_biored_train_data_fp),
                                                   "bc8_biored_task1_val_revealed.json")
    bc8_biored_val_revealed_data = {'source': 'PubMed', 'date': '2024-02-26, 19:00 (Portugal)', 'key': '', 'documents': []}

    def equal_entities(e1, e2):
        if e1["infons"] != e2["infons"]:
            return False
        if e1["text"] != e2["text"]:
            return False
        if e1["locations"] != e2["locations"]:
            return False
        #
        return True
    
    
    def get_the_document_from_the_original_biored_dataset(title):
        for data in biored_subsets:
            for doc in data['documents']:
                current_title = doc['passages'][0]['text']
                if current_title == title:
                    return doc
        assert False, 'No document was found given the title:\n{}\n'
    
    title_to_subset_and_pmid = dict()

    for i in range(3):
        subset = biored_subsets[i]
        subset_name = biored_subsets_names[i]
        for doc in subset['documents']:
            title = doc['passages'][0]['text']
            pmid = doc['id']
            assert title not in title_to_subset_and_pmid
            title_to_subset_and_pmid[title] = (subset_name, pmid)
    
    
    entity_types = set()
    relation_types = set()

    entities = dict()
    relations = dict()
    novel_relations = dict()

    entity_types_per_subset = {subset: set() for subset in bc8_biored_subsets_names}
    relation_types_per_subset = {subset: set() for subset in bc8_biored_subsets_names}

    entities_per_subset = {subset: dict() for subset in bc8_biored_subsets_names}
    relations_per_subset = {subset: dict() for subset in bc8_biored_subsets_names}
    novel_relations_per_subset = {subset: dict() for subset in bc8_biored_subsets_names}

    ignore_fake_val_docs = True
    # ignore_fake_val_docs = False

    for subset, subset_name in zip(bc8_biored_subsets, bc8_biored_subsets_names):
        for doc in subset['documents']:
            #
            if (subset_name == 'BC8_BioRED_val') and ignore_fake_val_docs:
                title = doc['passages'][0]['text']
                if title not in title_to_subset_and_pmid:
                    continue
            #
            entity_id_to_type = dict()
            for passage in doc['passages']:
                assert len(passage['relations']) == 0
                for annotation in passage['annotations']:
                    entity_type = annotation['infons']['type']
                    entity_ids = annotation['infons']['identifier'].split(',')
                    #
                    for entity_id in entity_ids:
                        if entity_id in entity_id_to_type:
                            assert entity_id_to_type[entity_id] == entity_type
                        else:
                            entity_id_to_type[entity_id] = entity_type
                    #
                    entity_types.add(entity_type)
                    entity_types_per_subset[subset_name].add(entity_type)
                    #
                    if entity_type not in entities:
                        entities[entity_type] = 0
                    entities[entity_type] += 1
                    #
                    if entity_type not in entities_per_subset[subset_name]:
                        entities_per_subset[subset_name][entity_type] = 0
                    entities_per_subset[subset_name][entity_type] += 1
            #
            for r in doc['relations']:
                r = r['infons']
                entity1 = r['entity1']
                entity2 = r['entity2']
                r_type = r['type']
                assert r['novel'] in ('Novel', 'No')
                r_is_novel = (r['novel'] == 'Novel')
                #
                entity1_type = entity_id_to_type[entity1]
                entity2_type = entity_id_to_type[entity2]
                #
                r_type_extended = '{}:{}->{}'.format(r_type, entity1_type, entity2_type)
                #
                relation_types.add(r_type_extended)
                relation_types_per_subset[subset_name].add(r_type_extended)
                #
                if r_type_extended not in relations:
                    relations[r_type_extended] = 0
                relations[r_type_extended] += 1
                #
                if r_type_extended not in relations_per_subset[subset_name]:
                    relations_per_subset[subset_name][r_type_extended] = 0
                relations_per_subset[subset_name][r_type_extended] += 1
                #
                if r_is_novel:
                    if r_type_extended not in novel_relations:
                        novel_relations[r_type_extended] = 0
                    novel_relations[r_type_extended] += 1
                #
                if r_is_novel:
                    if r_type_extended not in novel_relations_per_subset[subset_name]:
                        novel_relations_per_subset[subset_name][r_type_extended] = 0
                    novel_relations_per_subset[subset_name][r_type_extended] += 1
    
    VALID_ENTITY_TYPES = set(entities_per_subset['BC8_BioRED_train'].keys())
    VALID_RELATION_TYPES = set(relations_per_subset['BC8_BioRED_train'].keys())

    true_val_docs_come_from = {subset: 0 for subset in biored_subsets_names}

    for bc8_biored_doc in bc8_biored_val_data['documents']:
        title = bc8_biored_doc['passages'][0]['text']
        if title in title_to_subset_and_pmid:
            subset, pmid = title_to_subset_and_pmid[title]
            true_val_docs_come_from[subset] += 1
            #
            assert len(bc8_biored_doc['infons']) == 0
            assert len(bc8_biored_doc['relations']) == 0
            #
            biored_doc = get_the_document_from_the_original_biored_dataset(title)
            #
            # Verifications between the "BC8_BioRED" and "BioRED" datasets.
            #
            assert len(bc8_biored_doc['passages']) == 2
            assert len(biored_doc['passages']) == 2
            #
            assert bc8_biored_doc['passages'][0]['offset'] == biored_doc['passages'][0]['offset']
            assert bc8_biored_doc['passages'][0]['text'] == biored_doc['passages'][0]['text']
            #
            assert bc8_biored_doc['passages'][1]['offset'] == biored_doc['passages'][1]['offset']
            assert bc8_biored_doc['passages'][1]['text'] == biored_doc['passages'][1]['text'].rstrip()
            #
            # Verify that the entities in the "BC8_BioRED" are also
            # present in the original "BioRED" dataset.
            # Remember that the original "BioRED" dataset has some extra
            # entities that have the normalization identifier "-".
            #
            entity_id_to_type = dict()
            for i in range(2):
                bc8_biored_passage = bc8_biored_doc['passages'][i]
                biored_passage = biored_doc['passages'][i]
                #
                for e1 in bc8_biored_passage['annotations']:
                    assert e1['infons']['type'] in VALID_ENTITY_TYPES
                    found = False
                    for e2 in biored_passage['annotations']:
                        if equal_entities(e1, e2):
                            found = True
                            break
                    assert found, 'An entity from the "BC8_BioRED" dataset was not found in the original "BioRED" dataset.'
                #
                for e in bc8_biored_passage['annotations']:
                    entity_type = e['infons']['type']
                    entity_ids = e['infons']['identifier'].split(',')
                    for entity_id in entity_ids:
                        if entity_id in entity_id_to_type:
                            assert entity_id_to_type[entity_id] == entity_type
                        else:
                            entity_id_to_type[entity_id] = entity_type
            #
            # Verify that the relations present in the
            # original "BioRED" dataset are valid.
            #
            for r in biored_doc['relations']:
                r = r['infons']
                entity1 = r['entity1']
                entity2 = r['entity2']
                r_type = r['type']
                assert r['novel'] in ('Novel', 'No')
                #
                entity1_type = entity_id_to_type[entity1]
                entity2_type = entity_id_to_type[entity2]
                #
                r_type_extended = '{}:{}->{}'.format(r_type, entity1_type, entity2_type)
                assert r_type_extended in VALID_RELATION_TYPES
            #
            new_doc = dict()
            new_doc['id'] = bc8_biored_doc['id']
            new_doc['from'] = subset
            new_doc['pmid'] = pmid
            new_doc['passages'] = bc8_biored_doc['passages']
            new_doc['relations'] = biored_doc['relations']
            #
            bc8_biored_val_revealed_data['documents'].append(new_doc)
            
            
    with open(bc8_biored_val_revealed_data_fp, mode='w', encoding='utf-8') as fp:
        _ = fp.write(json.dumps(bc8_biored_val_revealed_data, indent=2))
        
    
def convert_tmvar3_corpus_to_tsv(tmvar_data_fp, tmvar_seqs_fp):
    
    VALID_ENTITY_TYPES = [
        'DNAMutation',
        'ProteinMutation',
        'SNP',
        'DNAAllele',
        'ProteinAllele',
        'AcidChange',
        'OtherMutation',
        'Gene',
        'Species',
        'CellLine',
    ]

    SEQVARIANT_ENTITY_TYPES = VALID_ENTITY_TYPES[:-3]
    
    ifp = open(tmvar_data_fp, mode='r', encoding='utf-8')
    ofp = open(tmvar_seqs_fp, mode='w', encoding='utf-8')


    _ = ofp.write('{}\t{}\t{}\t{}\n'.format('pmid', 'type', 'mention', 'identifier'))

    for line in ifp:
        line = line.strip()
        if re.match(r'\d+\|t\|', line):
            pmid, _, title = line.split('|')
            text = title
        elif re.match(r'\d+\|a\|', line):
            same_pmid, _, abstract = line.split('|')
            assert pmid == same_pmid
            text += ' '+ abstract
        elif line.count('\t') == 5:
            same_pmid, s, e, mention, e_type, e_id = line.split('\t')
            s = int(s)
            e = int(e)
            assert pmid == same_pmid
            #
            # Stupid offsets are incorrect in PMID 21904390.
            #
            assert (mention == text[s:e]) or (mention == text[s-1:e-1]) or (mention == text[s-2:e-2]) or \
                (mention == text[s-3:e-3]) or (mention == text[s-4:e-4]) or (mention == text[s-5:e-5]) or \
                (mention == text[s-6:e-6])
            assert e_type in VALID_ENTITY_TYPES
            if e_type in SEQVARIANT_ENTITY_TYPES:
                _ = ofp.write('{}\t{}\t{}\t{}\n'.format(pmid, e_type, mention, e_id))
        else:
            assert len(line) == 0


    ifp.close()
    ofp.close()
        

def list_files(directory):
    return [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files]


def check_if_files_exist(folder_path, files_to_check):
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"The {folder_path} is not a folder, it must be a folder")
    
    
    all_files_in_folder = set(list_files(folder_path))
    if "dataset" in folder_path:
        print("all_files_in_folder", all_files_in_folder)
    missing_files = files_to_check - all_files_in_folder

    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)

    return len(missing_files) == 0

def maybe_download_dataset(dataset_path):
    
    if check_if_files_exist(dataset_path, files_to_check={
        os.path.join(dataset_path,"bc8_biored_task1_train.json"),
        os.path.join(dataset_path,"bc8_biored_task1_val_revealed.json"),
    }):
        print("Found all of the dataset files")
    else:
        print("Some dataset files missing (see above), proceeding to download")
        
        download_and_unzip_file("https://ftp.ncbi.nlm.nih.gov/pub/lu/BC8-BioRED-track/BC8_BioRED_Subtask1_BioCJSON.zip",
                                os.path.join(dataset_path,"BC8_BioRED_Subtask1_BioCJSON.zip"))
        
        download_and_unzip_file("https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip",
                                os.path.join(dataset_path,"BIORED.zip"))
        
        convert_train_val_files(os.path.join(dataset_path, "bc8_biored_task1_train.json"),
                                os.path.join(dataset_path, "bc8_biored_task1_val.json"),
                                os.path.join(dataset_path, "BioRED/Train.BioC.JSON"),
                                os.path.join(dataset_path, "BioRED/Dev.BioC.JSON"),
                                os.path.join(dataset_path, "BioRED/Test.BioC.JSON"))    

def maybe_download_kb(kb_path):
    if check_if_files_exist(kb_path, files_to_check={
        os.path.join(kb_path,"Cellosaurus/concepts.jsonl"),
        os.path.join(kb_path,"Cellosaurus/concepts_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"CTD-diseases/concepts.jsonl"),
        os.path.join(kb_path,"CTD-diseases/concepts_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"CTD-diseases/definitions.jsonl"),
        os.path.join(kb_path,"CTD-diseases/definitions_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"CTD-diseases/synonyms.jsonl"),
        os.path.join(kb_path,"CTD-diseases/synonyms_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"dbSNP/tmVar3/seqvariants.tsv"),
        os.path.join(kb_path,"MeSH/concepts.jsonl"),
        os.path.join(kb_path,"MeSH/concepts_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"MeSH/concepts-supp.jsonl"),
        os.path.join(kb_path,"MeSH/concepts-supp_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"MeSH/definitions.jsonl"),
        os.path.join(kb_path,"MeSH/definitions_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"MeSH/definitions-supp.jsonl"),
        os.path.join(kb_path,"MeSH/definitions-supp_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"MeSH/synonyms.jsonl"),
        os.path.join(kb_path,"MeSH/synonyms_True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/3702_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/3702__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/7955_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/7955__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/9606_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/9606__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/10090_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/10090__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/10116_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/10116__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/11676_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/11676__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/12814_.jsonl"),
        os.path.join(kb_path,"NCBI-Gene/embeddings/12814__True_embeddings_sapBERT-multilanguage-large.npy"),
        os.path.join(kb_path,"NCBI-Gene/genes_with_tax.pickle"),
        os.path.join(kb_path,"NCBI-Gene/gene_lookup.json"),
        os.path.join(kb_path,"NCBI-Taxonomy/names.jsonl"),
        os.path.join(kb_path,"NCBI-Taxonomy/names_True_embeddings_sapBERT-multilanguage-large.npy"),
    }):
        print("Found all of the kb files")
    else:
        print("Some KB files missing (see above), proceeding to download")
        
        os.makedirs(os.path.join(kb_path,"Cellosaurus"), exist_ok=True)
        os.makedirs(os.path.join(kb_path,"CTD-diseases"), exist_ok=True)
        os.makedirs(os.path.join(kb_path,"dbSNP/tmVar3"), exist_ok=True)
        os.makedirs(os.path.join(kb_path,"MeSH"), exist_ok=True)
        os.makedirs(os.path.join(kb_path,"NCBI-Gene/embeddings"), exist_ok=True)
        os.makedirs(os.path.join(kb_path,"NCBI-Taxonomy"), exist_ok=True)
    
    
        # URL for the text file
        text_file_url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3Corpus.txt"
        download_file(text_file_url, os.path.join(kb_path,"dbSNP/tmVar3/tmVar3Corpus.txt"))
        convert_tmvar3_corpus_to_tsv(os.path.join(kb_path,"dbSNP/tmVar3/tmVar3Corpus.txt"),
                                     os.path.join(kb_path,"dbSNP/tmVar3/seqvariants.tsv"))
        
        
        # URL for the zipped file archive
        download_and_unzip_file("https://zenodo.org/records/11126786/files/Cellosaurus.zip?download=1",
                                os.path.join(kb_path,"Cellosaurus.zip"))
        download_and_unzip_file("https://zenodo.org/records/11126786/files/CTD-diseases.zip?download=1",
                                os.path.join(kb_path,"CTD-diseases.zip"))
        download_and_unzip_file("https://zenodo.org/records/11126786/files/MeSH.zip?download=1",
                                os.path.join(kb_path,"MeSH.zip"))
        download_and_unzip_file("https://zenodo.org/records/11204409/files/NCBI-Gene.zip?download=1",
                                os.path.join(kb_path,"NCBI-Gene.zip"))
        download_and_unzip_file("https://zenodo.org/records/11126786/files/NCBI-Taxonomy.zip?download=1",
                                os.path.join(kb_path,"NCBI-Taxonomy.zip"))
