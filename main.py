
import argparse
from src import grouping_args
from src.utils import load_biocjson


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source_file',
                        type=str,
                        help='')
    
    configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
    
    configs.add_argument('-t', '--tagger', dest="use_tagger", default=False, action='store_true', \
                         help='(default: False)')
    configs.add_argument('-l', '--linker', dest="use_linker", default=False, action='store_true', \
                            help='(default: False)')
    configs.add_argument('-e', '--extractor', dest="use_extractor", default=False, action='store_true', \
                            help='(default: False)')
    
    ### tagger options
    tagger_configs = parser.add_argument_group('Tagger settings', 'This settings are related to the indexer module.')
    tagger_configs.add_argument('--tagger.checkpoint', \
                                 type=str, default="trained_models/tagger/BioLinkBERT-large-dense-60-2-unk-P0.25-0.75-42-full/checkpoint-1200", \
                                 help='')
    tagger_configs.add_argument('--tagger.batch_size', \
                                 type=int, default=8, \
                                 help='')
    tagger_configs.add_argument('--tagger.output_file', \
                                 type=str, default="outputs/tagger/predicts.json", \
                                 help='')
    
    ## linker options
    linker_configs = parser.add_argument_group('Linker settings', 'This settings are related to the normalizer module.')
    linker_configs.add_argument('--linker.llm_api.module', \
                                 default=None, \
                                 help='')
    linker_configs.add_argument('--linker.llm_api.address', \
                                 default=None, \
                                 help='')
    linker_configs.add_argument('--linker.kb_folder', \
                                 default="knowledge-bases/", \
                                 help='')
    linker_configs.add_argument('--linker.dataset_folder', \
                                 default="dataset/", \
                                 help='')
    linker_configs.add_argument('--linker.output_file', \
                                 default="outputs/linker/predicts.json", \
                                 help='')
    
    # extractor options
    extractor_configs = parser.add_argument_group('Extractor settings', 'This settings are related to the extractor module.')
    extractor_configs.add_argument('--extractor.output_file', \
                                 type=str, default="outputs/extractor/predicts.json", \
                                 help='The extractor outputs path')
    extractor_configs.add_argument('--extractor.checkpoint', \
                                 type=str, default="trained_models/extractor/biolinkbert-large-full-mha-both-3-32456-20-mask-False/checkpoint-17340", \
                                 help='')
    extractor_configs.add_argument('--extractor.batch_size', \
                                 type=int, default=128, \
                                 help='')
    

    args = grouping_args(parser.parse_args())
    
    print(args)
    if not args.use_tagger and \
       not args.use_linker and \
       not args.use_extractor:
        # by default lets assume that we want to run the full pipeline!
        args.use_tagger, args.use_linker, args.use_extractor = True, True, True
    
    if (args.use_tagger, args.use_linker, args.use_extractor) in {(True, False, True)}:
        print("It is not possible to run the extractor after the annotator module in this pipeline. Any other configuration is valid. ")
        exit()
    
    pipeline = []
    
    if args.use_tagger:
        from src.tagger import Tagger
        pipeline.append(Tagger(**args.tagger.get_kwargs()))
    
    if args.use_linker:
        from src.linker import Linker
        pipeline.append(Linker(**args.linker.get_kwargs()))
        
    if args.use_extractor:
        from src.extractor import Extractor
        pipeline.append(Extractor(**args.extractor.get_kwargs()))
        
    print("Pipeline built")
    print(pipeline)
    
    print("Running")
    input_file = args.source_file
    for module in pipeline:
        input_file = module.run(input_file)
    
        
    