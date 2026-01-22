import torch
import math
import matplotlib.pyplot as plt

def tokenize_input(tokenizer, input_sentence):
    """Tokenize the input sentence."""

    tokenizer.truncation_side = "right" 
    inputs = tokenizer(input_sentence, return_tensors="pt", truncation=True, padding=True) # Add a [CLS] token at the beginning and a [SEP] token at the end of the sentence
    return inputs

def get_model_pred(logits):
    """Return the model prediction."""

    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs).item()
    return pred

def plot_attention_weights(attentions, tokens, layer=0):
    """Plot the attention weights for all heads in a given layer."""

    attention = attentions[layer][0].detach().cpu().numpy() # Select first batch element, shape (num_heads, seq_len, seq_len)
    n_heads = attention.shape[0]

    seq_len = len(tokens)

    # Determine subplot grid based on the number of heads 
    cols = int(math.ceil(math.sqrt(n_heads)))
    rows = int(math.ceil(n_heads / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    vmin, vmax = 0.0, 1.0 # Attention weights are softmax normalized

    for i in range(n_heads):
        ax = axes[i]

        # Plot attention heatmap 
        im = ax.imshow(attention[i], vmin=vmin, vmax=vmax, cmap='viridis')

        ax.set_title(f"Head {i}", fontsize=10)

        # Set token labels on x and y axis
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        ax.set_xlabel("Key", fontsize=9)
        ax.set_ylabel("Query", fontsize=9)

    # Hide any unused subplots
    for j in range(n_heads, len(axes)):
        axes[j].axis('off')

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(f"Attention heads at layer {layer}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()