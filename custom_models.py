import torch
import torch.nn as nn
import os # Added for path operations
import json # Added for saving config


### for ia3
class IA3Scaling(nn.Module):
    def __init__(self, base_linear, device):
        super().__init__()
        self.base = base_linear  # original nn.Linear
        # Scaling vector matches output dimension        
        # Wrap alpha in a Linear layer for DP compatibility
        self.ia3_alpha = nn.Linear(base_linear.out_features, base_linear.out_features, bias=False, device=device)
        # Initialize weights to identity matrix (ones on diagonal)
        torch.nn.init.constant_(self.ia3_alpha.weight, 0) # Initialize with zeros first
        with torch.no_grad():
            # Set diagonal elements to 1
            self.ia3_alpha.weight.copy_(torch.diag(torch.ones(base_linear.out_features, device=device)))

    def forward(self, x):
        out = self.base(x)
        # out = out * self.ia3_alpha  # Element-wise scaling (original idea)
        out = self.ia3_alpha(out)  # Element-wise scaling via diagonal linear layer
        return out

@torch.no_grad()
def clamp_to_diagonal(weight):
    with torch.no_grad():
        W = weight.data
        mask = torch.eye(W.size(0), device=W.device)
        W *= mask

def wrap_bert_for_ia3(model):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if "attention.self.key" in name or "attention.self.value" in name or "intermediate.dense" in name:
            parent = get_parent_module(model, name)
            attr = name.split('.')[-1]
            original = getattr(parent, attr)
            setattr(parent, attr, IA3Scaling(original, device))
    return model

def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent
### end ia3


### for soft prompt
class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, embedding_dim, device):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim, device=device))

    def forward(self, batch_size):
        # Expand prompt for the batch
        return self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
    
    
class SoftPromptModel(nn.Module):
    def __init__(self, base_model, prompt_length=10):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length

        embedding_dim = base_model.bert.embeddings.word_embeddings.embedding_dim
        model_device = next(base_model.parameters()).device
        self.soft_prompt_module = SoftPromptEmbedding(prompt_length, embedding_dim, model_device)

        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Make soft prompt trainable.
        self.trainable_params = (
            list(self.soft_prompt_module.parameters()) +
            list(self.base_model.classifier.parameters())
        )

        # Set requires_grad for the selected trainable parameters.
        for param in self.trainable_params:
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, **kwargs):

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            batch_size = input_ids.shape[0]
            input_embeds = self.base_model.bert.embeddings.word_embeddings(input_ids)
        else:
            batch_size = inputs_embeds.shape[0]
            input_embeds = inputs_embeds

        # STEP 1: Expand soft prompts for batch
        expanded_prompt = self.soft_prompt_module.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # STEP 2: Assign grad_sample directly to soft_prompt (required for DP optimization)
        def capture_grad_sample(grad):
            self.soft_prompt_module.soft_prompt.grad_sample = grad.detach().clone()

        expanded_prompt.register_hook(capture_grad_sample)

        # STEP 3: Concatenate prompt and input embeddings
        final_inputs_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)

        # STEP 4: Extend attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.prompt_length,
                                     dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask[:, :input_embeds.shape[1]]], dim=1)

        if token_type_ids is not None:
            prompt_token_types = torch.zeros(batch_size, self.prompt_length,
                                             dtype=token_type_ids.dtype,
                                             device=token_type_ids.device)
            token_type_ids = torch.cat([prompt_token_types, token_type_ids[:, :input_embeds.shape[1]]], dim=1)

        return self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=final_inputs_embeds,
            **kwargs
        )

    def save_pretrained(self, save_directory):
        """Saves the soft prompt and classifier state dictionary."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the soft prompt state dictionary
        soft_prompt_path = os.path.join(save_directory, "soft_prompt.pt")
        torch.save(self.soft_prompt_module.state_dict(), soft_prompt_path)

        # Save the classifier state dictionary
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.base_model.classifier.state_dict(), classifier_path)

        # Save configuration (e.g., prompt_length)
        config = {"prompt_length": self.prompt_length}
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
### end soft prompt


### for prefix tuning
class PrefixEmbedding(nn.Module):
    def __init__(self, prefix_length, hidden_dim, device):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(prefix_length, hidden_dim, device=device))

    def forward(self, batch_size):
        expanded = self.prefix.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, prefix_len, hidden_dim]

        # Attach hook to output, not parameter, to capture *per-sample* gradients
        def _capture_grad_sample(grad):
            if grad.dim() != 3:
                raise RuntimeError(f"Expected [B, P, D], got: {grad.shape}")
            self.prefix.grad_sample = grad.detach()

        expanded.register_hook(_capture_grad_sample)
        return expanded

class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length=10):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length

        # Access encoder layers (adapt for different architectures if needed)
        self.encoder_layers = base_model.bert.encoder.layer
        self.num_layers = len(self.encoder_layers)
        hidden_size = base_model.config.hidden_size
        device = next(base_model.parameters()).device

        # Create separate prefix for each layer
        self.prefix_modules = nn.ModuleList([
            PrefixEmbedding(prefix_length, hidden_size, device)
            for _ in range(self.num_layers)
        ])

        self._register_prefix_hooks()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Enable training for prefixes and classifier
        self.trainable_params = list(self.prefix_modules.parameters()) + list(self.base_model.classifier.parameters())
        for param in self.trainable_params:
            param.requires_grad = True

    def _register_prefix_hooks(self):
        def prepend_prefix_hook(layer_id):
            def hook(module, input):
                hidden_states = input[0]  # [B, T, D]
                batch_size = hidden_states.size(0)
                prefix = self.prefix_modules[layer_id](batch_size)  # [B, P, D]
                new_hidden_states = torch.cat([prefix, hidden_states], dim=1)
                return (new_hidden_states,)
            return hook

        for i, layer in enumerate(self.encoder_layers):
            layer.register_forward_pre_hook(prepend_prefix_hook(i))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        for i, prefix in enumerate(self.prefix_modules):
            torch.save(prefix.state_dict(), os.path.join(save_directory, f"prefix_layer_{i}.pt"))

        torch.save(self.base_model.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump({
                "prefix_length": self.prefix_length,
                "num_layers": self.num_layers
            }, f)
### end prefix tuning