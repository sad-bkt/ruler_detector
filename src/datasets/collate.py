import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if not dataset_items:
        return {}

    first_item = dataset_items[0]

    # Detection-style items: image tensor + target dict
    if "image" in first_item and "target" in first_item:
        batch = {
            "images": [item["image"] for item in dataset_items],
            "targets": [item["target"] for item in dataset_items],
        }
        if "image_id" in first_item:
            batch["image_ids"] = [item["image_id"] for item in dataset_items]
        if "path" in first_item:
            batch["paths"] = [item["path"] for item in dataset_items]
        return batch

    # default behaviour (classification example)
    result_batch = {}
    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
