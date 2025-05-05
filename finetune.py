from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
import evaluate
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
# Add PEFT library imports for LoRA, Prefix Tuning, and Prompt Tuning
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, PromptTuningConfig, PeftMixedModel

# Import manual IA3 wrapper from custom_models
import private_transformers.privacy_engine as pe_module
import logging

import warnings
# Suppress the specific FutureWarning about non-full backward hooks on Modules
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Using non-full backward hooks on a Module that does not return a single Tensor or a tuple of Tensors is deprecated"
)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune models with optional differential privacy')
parser.add_argument('--finetune_mode', type=str, default='full', 
                    choices=['full', 'last-layer', 'lora', 'ia3', 'prefix', 'soft-prompt', 'soft-prompt+lora', 'prefix+lora'], 
                    help='Fine-tuning mode: "full" for all layers, "last-layer" for classifier only, "lora" for LoRA, "ia3" for IA3 method, "prefix" for Prefix Tuning, "soft-prompt" for Soft-prompt Tuning, "soft-prompt+lora" for combined Soft-prompt and LoRA, or "prefix+lora" for combined Prefix Tuning and LoRA')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
# Differential Privacy arguments
parser.add_argument('--enable_dp', action='store_true', help='Enable differential privacy during training')
parser.add_argument('--target_epsilon', type=float, default=8.0, help='Target privacy budget epsilon for DP training')
parser.add_argument('--max_grad_norm', type=float, default=0.1, help='Maximum L2 norm of per-sample gradients for DP training')
# Dataset arguments
parser.add_argument('--datasets', nargs='+', default=["sst2", "qnli", "qqp", "mnli"],
                   choices=["sst2", "qnli", "qqp", "mnli"], 
                   help='List of datasets to train on, space separated (e.g., --datasets sst2 qnli)')
# LoRA arguments
parser.add_argument('--lora_r', type=int, default=8, help='Rank of the LoRA matrices')
parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')
# Prefix/Prompt Tuning arguments
parser.add_argument('--num_prepend_tokens', type=int, default=10, help='Number of tokens to prepend for Prefix/Soft-prompt Tuning')
args = parser.parse_args()

# Model configuration
model_checkpoint = "prajjwal1/bert-tiny"
max_length = 128

batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
finetune_mode = args.finetune_mode
enable_dp = args.enable_dp

# DP configuration
dp_config = {
    'target_epsilon': args.target_epsilon,
    'max_grad_norm': args.max_grad_norm,
}

# LoRA configuration
lora_config = {
    'r': args.lora_r,
    'alpha': args.lora_alpha,
    'dropout': args.lora_dropout,
}

# Prefix/Soft-prompt Tuning configuration
prefix_prompt_config = {
    'num_prepend_tokens': args.num_prepend_tokens,
}

# Dataset configuration
datasets_to_train = args.datasets
dataset_to_columns = {
    "sst2": {"sentence": "sentence", "label": "label"},
    "qnli": {"sentence1": "question", "sentence2": "sentence", "label": "label"},
    "mnli": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"},
    "qqp": {"sentence1": "question1", "sentence2": "question2", "label": "label"}
}

# Define tokenization function
def tokenize_function(examples, task):
    # Handle different input formats for different tasks
    if task in ["sst2"]:
        # Single sentence tasks
        return tokenizer(
            examples[dataset_to_columns[task]["sentence"]],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    else:
        # Sentence pair tasks
        return tokenizer(
            examples[dataset_to_columns[task]["sentence1"]],
            examples[dataset_to_columns[task]["sentence2"]],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

# Evaluation function
@torch.no_grad()
def evaluate_model(model, eval_dataloader, device, num_labels):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        outputs = model(**batch)
        logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=-1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    metric = evaluate.load("accuracy")
    result = metric.compute(predictions=all_preds, references=all_labels)
    
    return result

# Main training function
def train_model_on_dataset(dataset_name):
    print(f">> Training on {dataset_name} dataset...",
          f"(ε={dp_config['target_epsilon'] if enable_dp else '∞'},",
          f"clip_grad={dp_config['max_grad_norm'] if enable_dp else '∞'})")
    print(f"   Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Load dataset
    dataset = load_dataset("glue", dataset_name)
    
    # Determine number of labels for the dataset
    num_labels = 3 if dataset_name == "mnli" else 2
    
    # Suppress the specific transformers warning about newly initialized weights
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Load model with appropriate number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Freeze all parameters first. Classification layer is always trainable
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    if finetune_mode == 'full':
        print(f"   Fine-tuning mode: {finetune_mode} - Training all parameters")
        for name, param in model.named_parameters():
            param.requires_grad = True

    elif finetune_mode == 'last-layer':
        print(f"   Fine-tuning mode: {finetune_mode} - Only training the classifier layer")

    elif finetune_mode == 'lora':
        print(f"   Fine-tuning mode: {finetune_mode} - Using LoRA with rank={lora_config['r']}, alpha={lora_config['alpha']}, dropout={lora_config['dropout']}")
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            target_modules=["query", "key", "value"],
            modules_to_save=["classifier"], # Will be set as trainable and saved in the final checkpoint
        )
        model = get_peft_model(model, peft_config)
        
    elif finetune_mode == 'prefix':
        print(f"   Fine-tuning mode: {finetune_mode} - Using Prefix Tuning with {prefix_prompt_config['num_prepend_tokens']} tokens")

        from custom_models import PrefixModel
        model = PrefixModel(model, prefix_length=prefix_prompt_config['num_prepend_tokens'])

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True

    elif finetune_mode == 'soft-prompt':
        print(f"   Fine-tuning mode: {finetune_mode} - Using Soft-prompt Tuning with {prefix_prompt_config['num_prepend_tokens']} tokens")

        from custom_models import SoftPromptModel
        model = SoftPromptModel(model, prompt_length=prefix_prompt_config['num_prepend_tokens'])

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True

    elif finetune_mode == 'soft-prompt+lora':
        print(f"   Fine-tuning mode: {finetune_mode} - Using",
              f"LoRA (r={lora_config['r']}, alpha={lora_config['alpha']})",
              f"+ Soft-prompt Tuning ({prefix_prompt_config['num_prepend_tokens']} tokens)")

        # 1. Apply LoRA first
        lora_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            target_modules=["query", "key", "value"],
        )
        model = get_peft_model(model, lora_peft_config) # Apply LoRA to the base model

        # 2. Wrap the LoRA-adapted model with SoftPromptModel
        from custom_models import SoftPromptModel
        model = SoftPromptModel(model, prompt_length=prefix_prompt_config['num_prepend_tokens'])

        # 3. Set trainable parameters
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            # Freeze the copy of classifier generated by LoRA.
            # If requires_grad is True, this will mess up PrivacyEngine.
            if 'original_module' in name:
                param.requires_grad = False

    elif finetune_mode == 'prefix+lora':
        print(f"   Fine-tuning mode: {finetune_mode} - Using",
              f"Prefix Tuning ({prefix_prompt_config['num_prepend_tokens']} tokens)",
              f"+ LoRA (r={lora_config['r']}, alpha={lora_config['alpha']})")
        
        # 1. Apply LoRA first
        lora_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            target_modules=["query", "key", "value"],
        )
        model = get_peft_model(model, lora_peft_config)
        
        # 2. Wrap the LoRA-adapted model with PrefixMorel
        from custom_models import PrefixModel
        model = PrefixModel(model, prefix_length=prefix_prompt_config['num_prepend_tokens'])

        # 3. Set trainable parameters
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            # Freeze the copy of classifier generated by LoRA.
            # If requires_grad is True, this will mess up PrivacyEngine.
            if 'original_module' in name:
                param.requires_grad = False
    
    elif finetune_mode == 'ia3':
        print(f"   Fine-tuning mode: {finetune_mode} - Using IA3")
        from custom_models import wrap_bert_for_ia3, clamp_to_diagonal
        model = wrap_bert_for_ia3(model)
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'ia3' in name:
                param.requires_grad = True

    else:
        # return error if finetune_mode is not recognized
        raise ValueError(f"Unknown fine-tuning mode: {finetune_mode}.")

    # Collect all trainable parameters *after* all modifications
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Calculate and print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    
    # Detailed breakdown for PEFT methods if needed (optional, can be simplified)
    if finetune_mode == 'full':
        print(f"   Trainable parameters: {trainable_param_count} ({trainable_param_count/total_params:.2%} of total)")
    else:
        classifier_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n)
        if finetune_mode in ['soft-prompt+lora', 'prefix+lora']:
            # Calculate prompt/prefix and LoRA params separately for combined modes
            lora_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n)
            prompt_prefix_params_count = trainable_param_count - classifier_params_count - lora_params_count
            
            print(f"   Total trainable params: {trainable_param_count} ({trainable_param_count/total_params:.2%} of total)")
            print(f"     = Prompt/Prefix: {prompt_prefix_params_count} ({prompt_prefix_params_count/total_params:.2%})")
            print(f"     + LoRA: {lora_params_count} ({lora_params_count/total_params:.2%})")
            print(f"     + Classifier: {classifier_params_count} ({classifier_params_count/total_params:.2%})")
        else:
            # Original calculation for single PEFT methods
            peft_params_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'classifier' not in n)
            print(f"   Total trainable params: {trainable_param_count} ({trainable_param_count/total_params:.2%} of total)",
                  f"= PEFT {peft_params_count} ({peft_params_count/total_params:.2%})",
                  f"+ Classifier {classifier_params_count} ({classifier_params_count/total_params:.2%})")

    # Tokenize datasetq
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, dataset_name),
        batched=True
    )
    
    # Prepare datasets with correct format
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in dataset["train"].column_names if col != "label"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"], 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_key = "validation" if dataset_name != "mnli" else "validation_matched"
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset[val_key], 
        batch_size=batch_size
    )
    
    # Create optimizer manually - only with trainable parameters
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Setup output directory with mode info in name
    output_dir = f"./logs/{dataset_name}_{finetune_mode}_{'dp_' if enable_dp else ''}{datetime.now().strftime('%y%m%d.%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize privacy engine if DP is enabled
    privacy_engine = None
    if enable_dp:
        # Add warning suppression for the FutureWarning about backward hooks
        import warnings
        warnings.filterwarnings(
            "ignore", 
            message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated"
        )

        # monkey-patch private_transformers package bypass unsupported models
        if type(model) not in pe_module.SUPPORTED_TRANSFORMERS:
            pe_module.SUPPORTED_TRANSFORMERS += (type(model),)

        privacy_engine = pe_module.PrivacyEngine(
            model,
            batch_size=batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            sample_size=len(tokenized_dataset["train"]),
            epochs=epochs,
            max_grad_norm=dp_config['max_grad_norm'],
            target_epsilon=dp_config['target_epsilon'],
        )
        privacy_engine.attach(optimizer)
    
    # Training loop
    best_accuracy = 0.0
    train_metrics = []
    
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training loop
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            # Regular training
            if not enable_dp:
                # Standard training approach
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = F.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
            else:
                # DP training approach following dp_training_example.py
                optimizer.zero_grad()
                outputs = model(**batch)
                logits = outputs.logits
                
                # Compute per-sample losses
                loss = F.cross_entropy(logits, labels, reduction="none")
                
                # Use optimizer.step with loss as keyword argument
                optimizer.step(loss=loss)
                
                # Calculate mean loss for reporting
                batch_loss = loss.mean().item()
            
            if 'ia3' in finetune_mode:
                for name, weights in model.named_parameters():
                    if 'ia3' in name:
                        clamp_to_diagonal(weights)
            
            total_loss += batch_loss
            num_batches += 1
            progress_bar.set_postfix({"loss": batch_loss})

        # Evaluate after each epoch
        avg_loss = total_loss / num_batches
        eval_results = evaluate_model(model, eval_dataloader, device, num_labels)
        accuracy = eval_results['accuracy']
        print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss:.4f}, val acc: {accuracy:.4f}", end="")
        
        # Save metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "eval_accuracy": accuracy
        }
        train_metrics.append(epoch_metrics)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f" (New best accuracy!)", end="")
            # Save model
            model_path = f"{output_dir}/best_model"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        print()
    
    # Save final model
    final_model_path = f"{output_dir}/final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model for {dataset_name} saved to {final_model_path}")
    
    # Final evaluation
    model.eval()
    final_eval_results = evaluate_model(model, eval_dataloader, device, num_labels)
    
    # Print final results
    print(f"\n===== {dataset_name.upper()} RESULTS =====")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"   Final accuracy: {final_eval_results['accuracy']:.4f}")
    print("=" * 30)
    print()
    
    final_eval_results["best_accuracy"] = best_accuracy
    return final_eval_results


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Train on all specified datasets
results = {}
for dataset_name in datasets_to_train:
    results[dataset_name] = train_model_on_dataset(dataset_name)

# Print summary of results with DP information
print("\n===== SUMMARY OF RESULTS =====")
print(f"Training with Differential Privacy: {enable_dp}",
      f"(ε={dp_config['target_epsilon'] if enable_dp else '∞'},",
      f"clip_grad={dp_config['max_grad_norm'] if enable_dp else '∞'})")
print(f"Fine-tuning mode: {finetune_mode}", end=" ")
if 'lora' in finetune_mode: # Check if 'lora' is part of the mode name
    print(f"(LoRA config: rank={lora_config['r']}, alpha={lora_config['alpha']}, dropout={lora_config['dropout']})", end=" ")
if 'prefix' in finetune_mode or 'soft-prompt' in finetune_mode: # Check if 'prefix' or 'soft-prompt' is part of the mode name
    print(f"(Prefix/Soft-prompt config: num_prepend_tokens={prefix_prompt_config['num_prepend_tokens']})", end=" ")
print()
print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
print("=" * 30)
for dataset_name, result in results.items():
    print(f"{dataset_name}: {result}")
