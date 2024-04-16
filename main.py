
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source_file',
                        type=str,
                        help='')
    
    configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
    
    configs.add_argument('-t', '--tagger', default=False, action='store_true', \
                         help='(default: False)')
    configs.add_argument('-l', '--linker', default=False, action='store_true', \
                            help='(default: False)')
    configs.add_argument('-e', '--extractor', default=False, action='store_true', \
                            help='(default: False)')
    
    ### tagger options
    tagger_configs = parser.add_argument_group('Tagger settings', 'This settings are related to the indexer module.')
    tagger_configs.add_argument('--tagger.model_checkpoint', dest='tagger_model_checkpoint', \
                                 type=str, nargs='+', default=None, \
                                 help='The tagger model cfg path')
    
    ## linker options
    linker_configs = parser.add_argument_group('Linker settings', 'This settings are related to the normalizer module.')
    linker_configs.add_argument('--linker.llm_api', dest='linker_llm_api', \
                                 default=None, \
                                 help='')
    
    # extractor options
    extractor_configs = parser.add_argument_group('Extractor settings', 'This settings are related to the extractor module.')
    extractor_configs.add_argument('--extractor.write_path', dest='indexer_write_path', \
                                 type=str, default=None, \
                                 help='The extractor outputs path')
    
    args = parser.parse_args()
    
    if not args.tagger and \
       not args.linker and \
       not args.extractor:
        # by default lets assume that we want to run the full pipeline!
        args.tagger, args.linker, args.extractor = True, True, True
    
    if (args.tagger, args.linker, args.extractor) in {(True, False, True)}:
        print("It is not possible to run the extractor after the annotator module in this pipeline. Any other configuration is valid. ")
        exit()
        
    