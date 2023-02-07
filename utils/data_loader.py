from torch_geometric.loader import DataLoader


def load_train(train_graph):
    train_loader = DataLoader(train_graph, batch_size=64, shuffle=True)
    return train_loader

def load_valid(valid_graph):
    valid_loader = DataLoader(valid_graph, batch_size=64, shuffle=False)
    return valid_loader

def load_test(test_graph):
    test_loader = DataLoader(test_graph, batch_size=64, shuffle=False)
    return test_loader

