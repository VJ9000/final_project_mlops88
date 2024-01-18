from mlops88_ezCNNs.data.make_dataset import check_dvc

def test_check_dvc():
    assert check_dvc() == True