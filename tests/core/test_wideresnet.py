import pytest
import torch

import archs.wideresnet


class TestWideresnet:
    @pytest.fixture
    def standard_input(self):
        batch_size = 16
        channel_size = 3
        image_size = 32
        return torch.randn(
            batch_size, channel_size, image_size, image_size, dtype=torch.float
        )

    def test__standard_input(self, standard_input):
        batch_size = standard_input.size(0)

        factories = [
            archs.wideresnet.wideresnet16,
            archs.wideresnet.wideresnet28,
            archs.wideresnet.wideresnet40,
        ]
        for factory in factories:
            arch = factory()
            assert arch(standard_input).shape == torch.Size([batch_size, 1000])