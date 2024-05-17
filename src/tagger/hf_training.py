import os
import argparse

from data import load_train_test_split, load_old_train_test_split, SelectModelInputs, BIOTagger, RandomlyUKNTokens, EvaluationDataCollator, RandomlyReplaceTokens
from transformers import TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification

from model.configuration_bionexttager import BioNExtTaggerConfig
from model.modeling_bionexttagger import BioNExtTaggerModel
from trainer import NERTrainer
from metrics import NERMetrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--percentage_tags", type=float, default=0.2)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--p_augmentation", type=float, default=0.5)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=42)


    args = parser.parse_args()
    
    if args.augmentation=="none":
        # backwards compatibility
        args.augmentation = None
        
    model_checkpoint = args.checkpoint
    
    name = model_checkpoint.split("/")[1]
    
    if args.augmentation is not None:
        model_out_name = f"{name}-{args.epochs}-{args.context}-{args.augmentation}-P{args.p_augmentation}-{args.percentage_tags}-{args.random_seed}"
    else:
        model_out_name = f"{name}-{args.epochs}-{args.context}-{args.random_seed}"
    
    training_args = TrainingArguments(output_dir=os.path.join("../../trained_models_ner", model_out_name),
                                        num_train_epochs=args.epochs,
                                        dataloader_num_workers=1,
                                        dataloader_pin_memory=True,
                                        per_device_train_batch_size=args.batch,
                                        #gradient_accumulation_steps= 2, # batch 16 - 32 -64
                                        per_device_eval_batch_size= args.batch*4,
                                        prediction_loss_only=False,
                                        logging_steps = 10,
                                        logging_first_step = True,
                                        logging_strategy = "steps",
                                        seed=args.random_seed,
                                        data_seed=args.random_seed,
                                        eval_steps=None,#0.1, # 
                                        save_steps=99999, # this is changed latter in the code below
                                        save_strategy="steps",
                                        save_total_limit=1,
                                        evaluation_strategy="steps",
                                        warmup_ratio = 0.1,
                                        learning_rate=2e-5,
                                        weight_decay=0.01,
                                        push_to_hub=False,
                                        report_to="none",
                                        fp16=True,
                                        fp16_full_eval=False)
    
    CONTEXT_SIZE = args.context
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512
    
    biotagger = BIOTagger()
    transforms = [biotagger, SelectModelInputs()]
    
    train_augmentation = None
    if args.augmentation:
        if args.augmentation=="unk": 
            print("Note: The trainer will use RandomlyUKNTokens augmentation")
            train_augmentation = [RandomlyUKNTokens(tokenizer=tokenizer, 
                                context_size=CONTEXT_SIZE,
                                prob_change=args.p_augmentation, 
                                percentage_changed_tags=args.percentage_tags)]
        elif args.augmentation=="random":
            print("Note: The trainer will use RandomlyReplaceTokens augmentation")
            train_augmentation = [RandomlyReplaceTokens(tokenizer=tokenizer, 
                                context_size=CONTEXT_SIZE,
                                prob_change=args.p_augmentation, 
                                percentage_changed_tags=args.percentage_tags)]
            
    train_ds, test_ds = load_old_train_test_split("../../dataset/",
                                          tokenizer=tokenizer,
                                          context_size=CONTEXT_SIZE,
                                          train_transformations=transforms,
                                          train_augmentations=train_augmentation,
                                          test_transformations=None)
    
    # update the eval steps
    training_args.eval_steps = len(train_ds)//training_args.per_device_train_batch_size*training_args.num_train_epochs//5
    print("STEPS", training_args.eval_steps)

    training_args.save_steps = training_args.eval_steps
    
    id2label = {0:"O", 
            1:"B-GeneOrGeneProduct", 2:"I-GeneOrGeneProduct",
            3:"B-DiseaseOrPhenotypicFeature", 4:"I-DiseaseOrPhenotypicFeature",
            5:"B-ChemicalEntity", 6:"I-ChemicalEntity",
            7:"B-SequenceVariant", 8:"I-SequenceVariant",
            9:"B-OrganismTaxon", 10:"I-OrganismTaxon",
            11:"B-CellLine", 12:"I-CellLine"}

    label2id = {v:k for k,v in id2label.items()}
    
    
    config = BioNExtTaggerConfig.from_pretrained(model_checkpoint,
                                                 id2label = id2label,
                                                 label2id = label2id,
                                                 augmentation = args.augmentation,
                                                 context_size = args.context,
                                                 percentage_tags = args.percentage_tags,
                                                 p_augmentation = args.p_augmentation,
                                                 freeze = False,
                                                 crf_reduction = "mean")

    model = BioNExtTaggerModel.from_pretrained(model_checkpoint, config=config)
    model.training_mode() # fix a stupid bug regarding weight inits
    
    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding="longest",
                                                        label_pad_token_id=tokenizer.pad_token_id),
        eval_data_collator=EvaluationDataCollator(tokenizer=tokenizer, 
                                                padding=True,
                                                label_pad_token_id=tokenizer.pad_token_id),
        compute_metrics=NERMetrics(tagger=biotagger, context_size=CONTEXT_SIZE)
    )
    
    trainer.train()