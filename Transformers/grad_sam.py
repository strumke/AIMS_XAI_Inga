import torch
import torch.nn.functional as F

from utils import get_model_pred

seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_sam(model, attentions, logits):
    """Returns the Grad-SAM score for all tokens in an input sequence."""

    model = model.to(device)

    n_heads = model.config.num_attention_heads 
    n_layers = model.config.num_hidden_layers
    
    attention_grads = get_grads(model, attentions, logits) # Dict with layer indices as keys and gradient tensors of shape (n_heads, seq_lenth, seq_length) as values 

    seq_len = attentions[0].size(-1) # CLS and SEP tokens are included

    layer_vectors = []
    
    for layer_idx, grad in attention_grads.items():
        relu_grad = F.relu(grad.to(device)) # Shape (n_heads, seq_len, seq_len)

        attention = attentions[layer_idx].squeeze().clone().to(device) # Shape (n_heads, seq_len, seq_len)

        hadamard_product = attention * relu_grad

        # Sum attention weights for each head (sum over columns)
        summed_columns = hadamard_product.sum(dim=-1, keepdim=False) # Shape (n_heads, seq_len)

        # Sum over all heads
        summed_heads = summed_columns.sum(dim=0, keepdim=False) # Shape (seq_len)

        layer_vectors.append(summed_heads)

    grad_sam_scores = sum(layer_vectors) # Sum over all layers. Shape (batch_size, seq_len) where CLS and SEP tokens are included in seq_len.
    grad_sam_scores = grad_sam_scores / (n_layers*n_heads*seq_len) 

    return grad_sam_scores


def get_grads_gradsam(model, attentions, logits): 
    """Return the gradient of the output with respect to the attention weights for all heads in all layers."""

    n_layers = model.config.num_hidden_layers

    pred = get_model_pred(logits)
    logits = logits.squeeze()

    attention_grads = {layer_idx : None for layer_idx in range(n_layers)} 

    # Retain gradients for attention matrices
    for att in attentions:
        att.retain_grad()

    model.zero_grad()

    # Backward from the model prediction 
    logits[pred].backward(retain_graph=True)  

    # Collect gradients for each layer
    for layer_idx, att in enumerate(attentions):
        grad = att.grad.squeeze().clone() # Shape (n_heads, seq_len, seq_len)
        attention_grads[layer_idx] = grad
    
    return attention_grads

