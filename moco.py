import torch
from torch import nn
from torchvision import models

class MoCo(nn.Module):
    """
    backbone: feature extractor (e.g., ResNet-50).
    dim: feature dimension (e.g., 128).
    K: queue size (number of negative keys).
    m: momentum for updating key encoder.
    T: softmax temperature.
    """
    def __init__(self, base_model, dim: int = 128, K: int = 65536, m: float = 0.999, T: float = 0.07, mlp: bool = False):
        super().__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_model(num_classes=dim)
        self.encoder_k = base_model(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        # keys: (N, dim), already normalized
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue) with wrap-around
        end = ptr + batch_size
        if end <= self.K:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:self.K] = keys[:first].T
            self.queue[:, 0:end % self.K] = keys[first:].T

        self.queue_ptr[0] = end % self.K

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> torch.Tensor:
        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # Compute logits
        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)

        # Apply temperature
        logits /= self.T

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits
        


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    feature_dim = 2048
    queue_size = 1024

    resnet50 = lambda **kwargs: models.resnet50(weights=None, **kwargs)
    model = MoCo(base_model=resnet50, dim=feature_dim, K=queue_size)
    im_q = torch.randn(batch_size, 3, 224, 224)
    im_k = torch.randn(batch_size, 3, 224, 224)
    logits = model(im_q, im_k)
    print(logits.shape)  # Should be (batch_size, 1 + queue_size)
