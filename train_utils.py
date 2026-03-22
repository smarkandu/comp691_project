import torch
from torch.utils.data import DataLoader

def load_best_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(
        f"Loaded best checkpoint from epoch {ckpt.get('epoch', 'NA')} "
        f"with val_mAP={ckpt.get('val_map', 'NA')}"
    )
    return model

def make_loaders(
    train_ds,
    val_ds=None,
    test_ds=None,
    batch_size=32,
    num_workers=4,
    collate_fn=None,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader