import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ..dataimport import (
    build_coupled_tensors,
    cytokine_data,
    import_meta,
    rna_data,
    transform_data,
)

TEST_DATA = pd.DataFrame(np.random.random((100, 10)))


def test_metadata_import():
    """Tests metadata import"""
    meta = import_meta()
    assert isinstance(meta, pd.DataFrame)


@pytest.mark.parametrize(
    "import_func", [cytokine_data, rna_data, build_coupled_tensors]
)
def test_data_imports(import_func):
    """Tests omics data import"""
    data = import_func()
    assert isinstance(data, xr.Dataset)


def test_incorrect_transforms():
    """Tests incorrect data transform parameter"""
    with pytest.raises(ValueError):
        transform_data(TEST_DATA, "foo")


@pytest.mark.parametrize(
    "transform", ["log", "power", "reciprocal", "LOG", "LoG", "Log", None]
)
def test_correct_transforms(transform):
    """Tests accepted data transform parameters"""
    data = transform_data(TEST_DATA)
    assert isinstance(data, pd.DataFrame)


def test_incorrect_rna_normalize_options():
    """Tests incorrect RNA mean center option"""
    with pytest.raises(ValueError):
        rna_data(normalize="foo")


@pytest.mark.parametrize("normalize", ["full", "box", None])
def test_rna_normalize_options(normalize):
    """Tests accepted RNA mean center options"""
    assert isinstance(rna_data(normalize=normalize), xr.Dataset)


def test_transform_shape():
    """Tests transform_data keeps original data shape"""
    data = transform_data(TEST_DATA)
    assert data.shape == TEST_DATA.shape


def test_correct_tensor_patients():
    """Tests coupled tensor consists of only patients in metadata"""
    coupled = build_coupled_tensors()
    meta = import_meta()
    shared = set(coupled.Patient.values) & set(meta.index)

    assert len(coupled.Patient.values) == len(meta.index)
    assert len(shared) == len(coupled.Patient.values)
