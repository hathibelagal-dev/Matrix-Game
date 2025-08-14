import torch
import torch.nn.functional as F

def custom_attention(q, k, v, causal=False, dropout_p=0.0):
    # Get shapes
    batch_size, seq_len_q, hidden_size = q.shape
    _, seq_len_k, _ = k.shape
    num_heads = 8  # Adjust based on your model (e.g., hidden_size // head_dim)
    head_dim = hidden_size // num_heads

    # Reshape for multi-head attention: (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
    q = q.view(batch_size, seq_len_q, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len_k, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len_k, num_heads, head_dim).transpose(1, 2)

    # Compute attention using PyTorch SDPA
    attn_output = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=None,  # Add if needed
        dropout_p=dropout_p,
        is_causal=causal
    )

    # Reshape back to (batch_size, seq_len_q, hidden_size)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, hidden_size)
    return attn_output