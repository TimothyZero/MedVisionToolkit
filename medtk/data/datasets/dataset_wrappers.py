class RepeatDataset:
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.pipeline = self.dataset.pipeline
        self.times = times
        self._ori_len = len(self.dataset)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += f'dataset={self.dataset.__repr__()}' + ','
        repr_str += f'times={self.times}' + ')'
        return repr_str

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len

    def setLatitude(self, val):
        self.dataset.pipeline.setLatitude(val)

    def getLatitude(self):
        return self.dataset.pipeline.getLatitude()