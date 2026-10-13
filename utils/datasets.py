import os
import bisect
from typing import Optional, Callable, Tuple, Any, List, Iterable

from torch.utils.data import Dataset, IterableDataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class ImageList(VisionDataset):
    """Dataset defined by an image-list text file."""

    def __init__(
        self,
        root: str,
        classes: List[str],
        data_list_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        self.samples = self.parse_data_file(data_list_file)
        self.targets = [sample[1] for sample in self.samples]
        self.classes = classes
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        data_list = []

        with open(file_name, "r") as f:
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = " ".join(split_line[:-1])

                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)

                target = int(target)
                data_list.append((path, target))

        return data_list

    @property
    def num_classes(self) -> int:
        return len(self.classes)


class MultipleDomainsDataset(Dataset):
    """Keep the same sample structure as the original implementation."""

    def __init__(
        self,
        domains: Iterable[Dataset],
        domain_names: Iterable[str],
        domain_ids
    ):
        super().__init__()

        self.datasets = self.domains = list(domains)
        assert len(self.domains) > 0

        for domain in self.domains:
            assert not isinstance(domain, IterableDataset)

        self.cumulative_sizes = self.cumsum(self.domains)
        self.domain_names = domain_names
        self.domain_ids = domain_ids

    @staticmethod
    def cumsum(sequence):
        result = []
        total = 0

        for dataset in sequence:
            total += len(dataset)
            result.append(total)

        return result

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(
            self.cumulative_sizes,
            idx
        )

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = (
                idx - self.cumulative_sizes[dataset_idx - 1]
            )

        return (
            self.domains[dataset_idx][sample_idx]
            + (self.domain_ids[dataset_idx],)
        )

def get_dataset(
    root,
    train_source_transform,
    val_transform,
    train_target_transform=None
):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    image_list_dir = os.path.join(root, "image_list")

    source_train_list = os.path.join(
        image_list_dir, "source_train.txt"
    )
    target_train_list = os.path.join(
        image_list_dir, "target_train.txt"
    )
    target_test_list = os.path.join(
        image_list_dir, "target_test.txt"
    )

    # Infer class labels from the list files
    labels = set()

    for list_file in [
        source_train_list,
        target_train_list,
        target_test_list,
    ]:
        with open(list_file, "r") as f:
            for line in f:
                split_line = line.split()
                if split_line:
                    labels.add(int(split_line[-1]))

    class_names = [str(label) for label in sorted(labels)]

    # Source training dataset
    source_dataset = ImageList(
        root=root,
        classes=class_names,
        data_list_file=source_train_list,
        transform=train_source_transform
    )

    train_source_dataset = MultipleDomainsDataset(
        [source_dataset],
        ["source"],
        [0]
    )

    # Target training dataset
    target_dataset = ImageList(
        root=root,
        classes=class_names,
        data_list_file=target_train_list,
        transform=train_target_transform
    )

    train_target_dataset = MultipleDomainsDataset(
        [target_dataset],
        ["target"],
        [1]
    )

    # Target test = validation, as in the original implementation
    target_test_dataset = ImageList(
        root=root,
        classes=class_names,
        data_list_file=target_test_list,
        transform=val_transform
    )

    val_dataset = MultipleDomainsDataset(
        [target_test_dataset],
        ["target"],
        [1]
    )

    test_dataset = val_dataset

    num_classes = len(class_names)

    return (
        train_source_dataset,
        train_target_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        class_names
    )