def test_public_loader_import_has_no_network_calls():
    import importlib
    m = importlib.import_module("src.options.data.public_data_loaders")
    assert m is not None