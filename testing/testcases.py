import pytest
import os
from feast import FeatureStore

@pytest.fixture
def store():
    store = FeatureStore(os.path.abspath(os.path.join(os.getcwd() , os.pardir , 'feature_store//feature_repo')))
    return store


def test_features(store):
    entity_rows=[{"house_id": 1}, {"house_id": 2}]
    features=[
        "house_features:area",
        "house_features:bedrooms",
        "house_features:mainroad"
    ]
    df = store.get_online_features(features, entity_rows).to_dict()
    assert "bedrooms" in df