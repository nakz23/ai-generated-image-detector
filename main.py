from utils.dataset import get_dataloaders

data_dir = "data/small_dataset"


train_loader, val_loader, test_loader = get_dataloaders(data_dir)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))
