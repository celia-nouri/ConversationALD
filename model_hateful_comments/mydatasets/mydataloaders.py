from .hateful_discussions import HatefulDiscussionsDatasetLoader, HatefulDiscPygDatasetLoader

def get_data_loaders(size='medium', validation=True):
    print(f"Building hateful discussions dataloaders: size = {size}, validation = {validation}")

    ds = HatefulDiscussionsDatasetLoader(size, validation)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader

def get_pyg_data_loaders(size='medium', validation=True, seed=0):
    print(f"Building hateful discussions dataloaders: size = {size}, validation = {validation}")

    ds = HatefulDiscPygDatasetLoader(size, validation, seed)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader