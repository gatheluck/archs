import pytest
import torch

import archs.utilities.attention


class TestMultiheadSelfAttention:
    @pytest.fixture
    def standard_input(self):
        batch_size = 16
        len_sequence = 24
        dim_latent = 32
        return torch.randn(batch_size, len_sequence, dim_latent)

    def test__forward(self, standard_input):
        b, n, d = standard_input.size()
        msa = archs.utilities.attention.MultiheadSelfAttention(dim=d)
        assert msa(standard_input).size() == torch.Size([b, n, d])

    def test__backward(self, standard_input):
        b, n, d = standard_input.size()
        standard_input.requires_grad_()

        msa = archs.utilities.attention.MultiheadSelfAttention(dim=d)
        puseudo_loss = msa(standard_input).sum()

        # before backward, grad should be None
        assert standard_input.grad is None
        puseudo_loss.backward()
        # after backward, grad should have some value.
        assert standard_input.grad is not None
