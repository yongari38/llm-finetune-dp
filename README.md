# LLM Fine-tuning with Differential Privacy
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/18GXgZSb1dQLMiZo8y-Kh8a8VmDoCAPMG?usp=copy)

This project provides a framework for fine-tuning LLMs (e.g., BERT) on various GLUE benchmark tasks using different fine-tuning methods and optional Differential Privacy (DP).

## Features

*   **Base Models**: Only tested on `prajjwal1/bert-tiny`.
*   **Datasets**: Fine-tunes on selected GLUE tasks:
    *   SST-2
    *   QNLI
    *   QQP
    *   MNLI
*   **Fine-tuning Modes**: Implements several fine-tuning strategies:
    *   `full`: Train all model parameters.
    *   `last-layer`: Train only the final classification layer.
    *   `lora`: Low-Rank Adaptation.
    *   `ia3`: (IA)^3 - Infused Adapter by Inhibiting and Amplifying Inner Activations.
    *   `prefix`: Prefix Tuning.
    *   `soft-prompt`: Soft Prompt Tuning (Prompt Tuning).
    *   `soft-prompt+lora`: Combination of Soft Prompt and LoRA.
    *   `prefix+lora`: Combination of Prefix Tuning and LoRA.
*   **Differential Privacy**: Integrates with `private-transformers` to enable DP training.

## Installation

```bash
conda create -n finetune python=3.10
conda activate finetune
pip install git+https://github.com/lxuechen/private-transformers.git peft scikit-learn evaluate
```

## Usage

Run `python finetune.py --help` to see all available options:

```
usage: finetune.py [-h] [--model_checkpoint MODEL_CHECKPOINT] [--max_length MAX_LENGTH]
                   [--finetune_mode {full,last-layer,lora,ia3,prefix,soft-prompt,soft-prompt+lora,prefix+lora}]
                   [--learning_rate LEARNING_RATE] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--enable_dp]
                   [--target_epsilon TARGET_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
                   [--datasets {sst2,qnli,qqp,mnli} [{sst2,qnli,qqp,mnli} ...]] [--lora_r LORA_R]
                   [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT]
                   [--num_prepend_tokens NUM_PREPEND_TOKENS]

Fine-tune models with optional differential privacy

options:
  -h, --help            show this help message and exit
  --model_checkpoint MODEL_CHECKPOINT
                        Base model checkpoint
  --max_length MAX_LENGTH
                        Max sequence length for tokenizer
  --finetune_mode {full,last-layer,lora,ia3,prefix,soft-prompt,soft-prompt+lora,prefix+lora}
                        Fine-tuning mode
  --learning_rate LEARNING_RATE
                        Learning rate
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size
  --enable_dp           Enable differential privacy
  --target_epsilon TARGET_EPSILON
                        Target epsilon for DP
  --max_grad_norm MAX_GRAD_NORM
                        Max grad norm for DP
  --datasets {sst2,qnli,qqp,mnli} [{sst2,qnli,qqp,mnli} ...]
                        List of datasets to train on
  --lora_r LORA_R       LoRA rank
  --lora_alpha LORA_ALPHA
                        LoRA alpha
  --lora_dropout LORA_DROPOUT
                        LoRA dropout
  --num_prepend_tokens NUM_PREPEND_TOKENS
                        Number of prepend tokens for Prefix/Soft-prompt
```

**Example1:** Full fine-tuning on SST-2 without DP:
```bash
python finetune.py \
    --datasets sst2 \
    --finetune_mode full
```

**Example2:** IA3 fine-tuning on QNLI and QQP with DP (ε=8.0, MaxGradNorm=0.1):
```bash
python finetune.py \
    --datasets qnli qqp \
    --finetune_mode ia3 \
    --enable_dp \
    --target_epsilon 8.0 \
    --max_grad_norm 0.1
```
