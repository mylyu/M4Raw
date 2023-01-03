# from .dataset import GoProDataset, MixDataset, VideoDataset
from .data_sampler import DistIterSampler
import torch
import torch.utils.data


def create_dataloader(dataset, args, sampler=None):
    phase = args.phase
    if phase == 'train':
        if args.dist:
            world_size = torch.distributed.get_world_size()
            num_workers = args.num_workers
            assert args.batch_size % world_size == 0
            batch_size = args.batch_size // world_size
            shuffle = False
        else:
            num_workers = args.num_workers * len(args.gpu_ids)
            batch_size = args.batch_size
            shuffle = True
        return MultiEpochsDataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)