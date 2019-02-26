import torch.utils.data as data


class MyDataset(data.Dataset):
    """
    customed dataset class
    """
    def __init__(self, images, labels, ids, timestep):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.timestep = timestep

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


class Forecast(data.Dataset):
    """
    customed dataset class
    """
    def __init__(self, M1errors, M2errors, ids, timestep):
        self.M1errors = M1errors
        self.M2errors = M2errors
        self.ids = ids
        self.timestep = timestep

    def __getitem__(self, index):
        M1errors, M2errors = self.M1errors[index], self.M2errors[index]
        return M1errors, M2errors

    def __len__(self):
        return len(self.M1errors)
