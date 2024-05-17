

from collections import ChainMap

import evaluate

from transformers import TrainingArguments, Trainer, AutoTokenizer

from transformers.trainer_utils import EvalPrediction

from data import load_train_test_split
from data import SelectModelInputs

from collator import DataCollatorForRelationAndNovelClassification
import argparse

from trainer import JointTrainer
from collator import DataCollatorForRelationClassification
import numpy as np
import evaluate
from model.modeling_bionextextractor import BioNExtExtractorModel
from model.configuration_bionextextractor import BioNExtExtractorConfig

def compute_metrics_relations(p: EvalPrediction):
    
        predictions = np.argmax(p.predictions, axis=-1)
        macro_metrics = dict(ChainMap(*[metric.compute(predictions=predictions, references=p.label_ids, average="macro") for metric in metrics]))
        micro_metrics = dict(ChainMap(*[metric.compute(predictions=predictions, references=p.label_ids, average="micro") for metric in metrics]))
    
        return {f"macro_relation_{k}":v for k, v in macro_metrics.items()} | {f"micro_relation{k}":v for k, v in micro_metrics.items()}

def compute_metrics_relations_and_novelty(p: EvalPrediction):
    relation_logits, novel_logits = p.predictions
    relation_labels = p.label_ids[:,0]
    novel_labels = p.label_ids[:,1]
    
    relation_preds = np.argmax(relation_logits, axis=-1)
    novel_preds = np.argmax(novel_logits, axis=-1)
    
    r_macro_metrics = dict(ChainMap(*[metric.compute(predictions=relation_preds, references=relation_labels, average="macro") for metric in metrics]))
    r_micro_metrics = dict(ChainMap(*[metric.compute(predictions=relation_preds, references=relation_labels, average="micro") for metric in metrics]))
    n_micro_metrics = dict(ChainMap(*[metric.compute(predictions=novel_preds, references=novel_labels, average="micro") for metric in metrics]))
    return {f"macro_relation_{k}":v for k, v in r_macro_metrics.items()} | {f"micro_relation{k}":v for k, v in r_micro_metrics.items()} | {f"micro_novel{k}":v for k, v in n_micro_metrics.items()} 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("checkpoint", type=str, default=None)
    parser.add_argument("--arch", type=str, default="mha")
    parser.add_argument("--index_type", type=str, default="both")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--novel", action="store_true")
    args = parser.parse_args()
    
    model_checkpoint = args.checkpoint
    
    name = model_checkpoint.split("/")[1]

    dir_name = f"../../trained_models_extrator/{name}-{args.batch}-{args.random_seed}"
    
    training_args = TrainingArguments(output_dir=dir_name,
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
                                        save_strategy="epoch",
                                        save_total_limit=1,
                                        evaluation_strategy="epoch",
                                        warmup_ratio = 0.1,
                                        learning_rate=2e-5,
                                        weight_decay=0.01,
                                        push_to_hub=False,
                                        report_to="none",
                                        fp16=True,
                                        fp16_full_eval=False)
        
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512

    special_tokens = ['[s1]','[e1]', '[s2]','[e2]' ]
    num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True) ##This line is updated

    if args.novel:
        select_keys = ["input_ids", "attention_mask", "label", "indexes", "novel"]
    else:
        select_keys = ["input_ids", "attention_mask", "label", "indexes"]

    transforms = [SelectModelInputs(select_keys)]

    train_ds, test_ds = load_train_test_split("../../dataset/",
                                            tokenizer=tokenizer,
                                                index_type=args.index_type,
                                                special_tokens=special_tokens,
                                            train_transformations=transforms,
                                            train_augmentations=None,
                                            test_transformations=transforms)

    label2id = {'Association': 0, 'Positive_Correlation':1, 'Negative_Correlation':2,
                'Cotreatment': 3, 'Bind': 4, 'Comparison': 5, 'Conversion': 6,    
                'Drug_Interaction':7, 'Negative_Class': 8}

    id2label = {v:k for k,v in label2id.items()}

    config = BioNExtExtractorConfig.from_pretrained(model_checkpoint,
                                                    id2label = id2label,
                                                    label2id = label2id,
                                                    arch_type = args.arch,
                                                    index_type = args.index_type,
                                                    novel = args.novel,
                                                    tokenizer_special_tokens=special_tokens,
                                                    update_vocab = len(tokenizer))


    model = BioNExtExtractorModel.from_pretrained(model_checkpoint, config=config)
    model.training_mode()
    
    metrics = [evaluate.load(m) for m in ["f1", "precision", "recall"]]
    
    trainer_class = JointTrainer if args.novel else Trainer
    
    trainer = trainer_class(
        model = model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForRelationAndNovelClassification(tokenizer=tokenizer, 
                                                                padding="longest"),
        compute_metrics= compute_metrics_relations_and_novelty if args.novel else compute_metrics_relations
        
    )
    
    print(trainer)
    trainer.train() 