import torchvision.transforms as transforms
from torchvision import datasets


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path

    def __len__(self) -> int:
        return super().__len__()


def transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    return transform


def search(annoy_index, query_vector, k=5):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    return indices
