import pytest
import torch

from unext import UNeXt

def test_smoke():
    unext = UNeXt(1, 3, widths=[32, 64, 128])
    input = torch.randn(2, 1, 128, 128)
    output = unext(input)

    assert output.shape == (2, 3, 128, 128)