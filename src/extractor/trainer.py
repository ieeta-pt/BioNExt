from transformers import Trainer
import torch

class JointTrainer(Trainer):
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):

        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Objective is to propagate the novel labels, so lets add them to the labels
        labels = torch.vstack([labels, inputs["novel"].to(labels.device)]).T
        
        return loss, logits, labels
    
    
    