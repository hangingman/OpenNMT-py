import torch
import torch.nn as nn


class SimpleFusion(nn.Module):

    def __init__(self, language_model, tgt_vocab, lm_vocab):
        super(SimpleFusion, self).__init__()

        self.lang_model = language_model.eval()

        # Remove the language model parameters from the parameters
        # to be optimized
        for param in self.lang_model.parameters():
            param.requires_grad = False

        self.lang_model.num_directions = 1
        fusion_idx_list = []
        for idx, token in enumerate(tgt_vocab.itos):
            if token in lm_vocab.itos:
                fusion_idx_list.append(idx)

        self.fusion_idxs = torch.tensor(
            fusion_idx_list,
            dtype=torch.long)

    def forward(self, char_inp, scores, lm_hidden_state=None):

        if len(scores.shape) == 3:
            scores = scores.view(scores.shape[0]*scores.shape[1], -1)
        else:
            char_inp.unsqueeze_(-1)

        self.lang_model = self.lang_model.eval()

        outputs, lm_hidden_state = self.lang_model(
            char_inp, None,
            lm_hidden_state)
        outputs = outputs[-1].contiguous()
        lm_log_probs, _ = self.lang_model.generator(outputs,
                                                    None)

        fusion_log_probs = scores.new_full(scores.shape, scores[:, 1].mean())

        fusion_log_probs[:, self.fusion_idxs] =\
            lm_log_probs.view(-1, lm_log_probs.shape[-1])

        fusion_log_probs = torch.log_softmax(fusion_log_probs, dim=-1)

        fusion_log_probs += scores

        return torch.log_softmax(fusion_log_probs, dim=-1), lm_hidden_state
