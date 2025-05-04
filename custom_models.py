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

        # STEP 2: Register hook that assigns grad_sample directly to the soft_prompt Parameter
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

class PrefixTuningModel(nn.Module):
    """Wrapper for models with prefix tuning.
    
    This adds N learnable prefix vectors to each transformer layer.
    Only these prefix vectors are trained, while the rest of the model is frozen.
    """
    
    def __init__(self, base_model, prefix_length=10):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        
        # Get key and value dimensions from the base model
        model_config = base_model.config
        self.n_layers = getattr(model_config, "num_hidden_layers", None)        
        self.hidden_size = getattr(model_config, "hidden_size", None)
        
        # Create prefix parameters for each layer
        # We need key and value prefix for each attention head in each layer
        self.prefix_key_values = nn.ParameterList()
        
        # For each layer, create prefix parameters for both keys and values
        for i in range(self.n_layers):
            # Create the parameter for this layer (keys and values)
            # Shape: [prefix_length, 2, hidden_size]
            # The 2 is for key and value
            layer_prefix = nn.Parameter(torch.randn(prefix_length, 2, self.hidden_size))
            self.prefix_key_values.append(layer_prefix)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Record which parameters should be optimized
        self.trainable_params = list(self.prefix_key_values)
            
        # Store the original forward functions for each layer to modify them
        self._store_original_forward_functions()
        
        # Replace the forward functions with our custom ones
        self._replace_forward_functions()
    
    def _store_original_forward_functions(self):
        """Store the original attention forward functions to restore later if needed."""
        self.original_forward_funcs = {}
        
        for i, layer in enumerate(self.base_model.bert.encoder.layer):
            self.original_forward_funcs[i] = layer.attention.self.forward
    
    def _replace_forward_functions(self):
        """Replace the forward functions of self-attention layers with our custom ones."""
        layers = self.base_model.bert.encoder.layer
        
        for i, layer in enumerate(layers):
            # Get the prefix parameters for this layer
            layer_prefix = self.prefix_key_values[i]
            
            # Define custom forward function with prefix
            def make_custom_forward(layer_idx, orig_forward):
                def custom_forward(self, *args, **kwargs):
                    # Call original forward function
                    outputs = orig_forward(self, *args, **kwargs)
                    
                    # Get batch size from hidden states
                    if 'hidden_states' in kwargs:
                        batch_size = kwargs['hidden_states'].size(0)
                    else:
                        batch_size = args[0].size(0)
                    
                    # Get prefix for this layer
                    prefix_params = layer_prefix
                    
                    # Expand for batch size
                    # From [prefix_length, 2, hidden_size] to [batch_size, prefix_length, 2, hidden_size]
                    expanded_prefix = prefix_params.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    
                    # Reshape to separate key and value
                    # [batch_size, prefix_length, 2, hidden_size] -> [2, batch_size, prefix_length, hidden_size]
                    expanded_prefix = expanded_prefix.permute(2, 0, 1, 3)
                    
                    # Now we have separate key and value prefixes
                    prefix_keys = expanded_prefix[0]  # [batch_size, prefix_length, hidden_size]
                    prefix_values = expanded_prefix[1]  # [batch_size, prefix_length, hidden_size]
                    
                    # Prepend to the key and value tensors
                    if isinstance(outputs, tuple):
                        # Most models return (attention_output, attention_weights)
                        key_states = outputs[0]  # [batch_size, seq_len, hidden_size]
                        value_states = outputs[1]  # [batch_size, seq_len, hidden_size]
                        
                        # Prepend prefix
                        key_states_with_prefix = torch.cat([prefix_keys, key_states], dim=1)
                        value_states_with_prefix = torch.cat([prefix_values, value_states], dim=1)
                        
                        return (key_states_with_prefix, value_states_with_prefix, *outputs[2:])
                    else:
                        # Return modified outputs
                        return outputs
                
                return custom_forward
            
            # Monkey-patch the forward function
            layer.attention.self.forward = make_custom_forward(i, self.original_forward_funcs[i])
    
    def forward(self, *args, **kwargs):
        """Forward pass, delegating to the base model after adjusting inputs for prefix."""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        """Returns the parameters that should be trained."""
        return self.trainable_params
