import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMAggregatorTorch(nn.Module):
    """
    Aggregatore LSTM puro PyTorch, per GNN custom.
    Prende i messaggi in ingresso raggruppati per nodo,
    li processa come sequenza e restituisce l'ultimo hidden state per ciascun nodo.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, messages, dst, num_nodes):
        """
        messages: Tensor [num_edges, msg_dim]
        dst: Tensor [num_edges], nodo ricevente per ciascun messaggio
        num_nodes: int, numero nodi nel grafo

        Output: [num_nodes, hidden_dim]
        """
        # 1. Raggruppa messaggi per nodo ricevente
        msgs_per_node = [[] for _ in range(num_nodes)]
        for i in range(dst.shape[0]):
            msgs_per_node[dst[i].item()].append(messages[i])
        # 2. Crea sequenze, pad per batch LSTM
        seq_tensors = [
            (
                torch.stack(msgs)
                if len(msgs) > 0
                else torch.zeros(1, messages.shape[1], device=messages.device)
            )
            for msgs in msgs_per_node
        ]
        lengths = [len(msgs) if len(msgs) > 0 else 1 for msgs in msgs_per_node]
        padded_seqs = pad_sequence(
            seq_tensors, batch_first=True
        )  # [num_nodes, max_seq_len, msg_dim]
        lengths = torch.tensor(lengths)
        # 3. LSTM
        packed_input = pack_padded_sequence(
            padded_seqs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hn, cn) = self.lstm(packed_input)
        agg, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 4. Prendi ultimo hidden state valido per ogni nodo
        agg_out = []
        for i, l in enumerate(lengths):
            agg_out.append(agg[i, l - 1, :])
        agg_out = torch.stack(agg_out, dim=0)  # [num_nodes, hidden_dim]
        return agg_out
