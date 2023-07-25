import torch
import torch.nn as nn
import torch.nn.functional as F

def self_attention(inputs, name):
    """
    :param inputs: input tensor (B, T, 3, dim)
    :param name: scope name
    :return: final_output (B, T, dim)
    """
    t = inputs.size(1)
    share_param = True
    hidden_size = inputs.size(-1)  # D value - hidden size of the RNN layer

    if share_param:
        scope_name = 'self_attn'
    else:
        scope_name = 'self_attn' + name

    inputs = inputs.transpose(1, 0)  # (T, B, 3, dim)

    outputs = []
    for x in range(t):
        t_x = inputs[x]  # (B, 3, dim)

        den = True
        if den:
            x_proj = nn.Linear(hidden_size, hidden_size)(t_x)
            x_proj = torch.tanh(x_proj)
        else:
            x_proj = t_x

        u_w = nn.Parameter(torch.randn(hidden_size, 1) * 0.01, requires_grad=True)
        x = torch.matmul(x_proj, u_w)  # (B, 3, 1)
        alphas = F.softmax(x, dim=1)  # (B, 3, 1)

        output = torch.matmul(t_x.transpose(1, 2), alphas)  # (B, dim, 1)
        output = output.squeeze(-1)  # (B, dim)
        outputs.append(output)

    final_output = torch.stack(outputs, dim=1)  # (B, T, dim)

    return final_output