import torch

class CrossEntropyLoss:
    def __init__(self,ignore_index=-1):
        self.ignore_index = ignore_index

    def __call__(self, logits, target):
        batch_size,seq_len,vocab_size = logits.shape
        logits_flat = logits.view(-1,vocab_size)
        target_flat = target.view(-1)

        log_probs = torch.nn.functional.log_softmax(logits_flat,dim=-1)

        # 计算log_softmax（数值稳定）

        # 负对数似然损失
        nll_loss = -log_probs.gather(-1,target_flat.unsqueeze(1)).squeeze(1)

        return nll_loss.mean()

