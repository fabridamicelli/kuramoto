import pytest
import numpy as np

from kuramoto import Kuramoto


def test_n_nodes_natfreqs_none():
    with pytest.raises(ValueError):
        Kuramoto(n_nodes=None, natfreqs=None)

def test_natfreqs():
    Kuramoto(n_nodes=None, natfreqs=np.array([1,2,3]))

def test_n_nodes():
    Kuramoto(n_nodes=3, natfreqs=np.array([1,2,3]))

def test_n_nodes():
    Kuramoto(n_nodes=3, natfreqs=None)
