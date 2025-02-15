from .conversation_dataset import ConversationDatasetLoader, ConversationGraphDatasetLoader

def get_dataloaders(size='medium', validation=True):
    print(f"Building ALD conversation dataloaders: size = {size}, validation = {validation}")

    ds = ConversationDatasetLoader(size, validation)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader

def get_graph_dataloaders(size='medium', validation=True, seed=0):
    print(f"Building ALD graph conversation dataloaders: size = {size}, validation = {validation}")

    ds = ConversationGraphDatasetLoader(size, validation, seed)
    train_loader, valid_loader, test_loader = ds.get_train_data(), ds.get_val_data(), ds.get_test_data()
    return train_loader, valid_loader, test_loader