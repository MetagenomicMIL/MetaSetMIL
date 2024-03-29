import logging

import torch
from base.base_dataloader import BaseDataLoader
from utils import transforms as mytransforms
from utils.util import get_global_rank, get_world_size

from . import datasets


class RefSeqProkaryotaDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 genome_dir,
                 taxonomy_dir,
                 total_samples,
                 batch_size,
                 fixed_dataset=False,
                 drop_last=False,
                 dataset_distribution='uniform',
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 sample_cache_size=0):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyRead(error_model,
                                                rmin,
                                                rmax,
                                                p=noise)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset

        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[1, -1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryota(
            genome_dir,
            taxonomy_dir,
            total_samples,
            accessions_file,
            taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            dataset_distribution=dataset_distribution,
            genome_cache_size=genome_cache_size)

        super(RefSeqProkaryotaDataLoader,
              self).__init__(self.dataset, batch_size, shuffle,
                             validation_split, num_workers, drop_last)

    def enable_multithreading_if_possible(self):
        if self.dataset.genome_cache_is_full():
            try:
                if self.num_workers != self._num_workers:
                    self.num_workers = self._num_workers
                    self.logger.info(f'Enabling {self.num_workers} '
                                     'workers for data loading...')
            except AttributeError:
                pass
        else:
            self._num_workers = self.num_workers
            self.num_workers = 0

    def step(self, epoch):
        super().step(epoch)
        self.enable_multithreading_if_possible()
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            seed = epoch
            seed = seed * get_world_size() + get_global_rank()
            if self.valid_loader:
                seed = 2**32 - seed
                self.dataset.set_to_lognorm()
            else:
                self.dataset.set_to_default_sim()
            self.dataset.simulator.set_seed(seed)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True


class RefSeqProkaryotaBagsDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 target_format,
                 genome_dir,
                 taxonomy_dir,
                 total_bags,
                 bag_size,
                 batch_size,
                 fixed_dataset=False,
                 drop_last=False,
                 dataset_distribution='uniform',
                 reseed_every_n_bags=1,
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyRead(error_model,
                                                rmin,
                                                rmax,
                                                p=noise)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset

        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[1, -1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryotaBags(
            bag_size=bag_size,
            total_bags=total_bags,
            target_format=target_format,
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            dataset_distribution=dataset_distribution,
            reseed_every_n_bags=reseed_every_n_bags,
            genome_cache_size=genome_cache_size,
            num_workers=num_workers,
            single_read_target_vectors=single_read_target_vectors)

        super(RefSeqProkaryotaBagsDataLoader,
              self).__init__(self.dataset, batch_size, shuffle,
                             validation_split, 0, drop_last)

    def step(self, epoch):
        super().step(epoch)
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            if self.valid_loader:
                self.dataset.set_to_lognorm()
            else:
                self.dataset.set_to_default_sim()

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True


class RefSeqProkaryotaKmerDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 genome_dir,
                 taxonomy_dir,
                 total_samples,
                 batch_size,
                 kmer_vocab_file,
                 fixed_dataset=False,
                 drop_last=False,
                 dataset_distribution='uniform',
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 forward_reads_only=False,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyKmerRead(
            kmer_vocab_file,
            error_model,
            rmin,
            rmax,
            p=noise,
            forward_reads_only=forward_reads_only)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset

        self.vocab_size = len(g2read.kmer2id)
        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[-1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryota(
            genome_dir,
            taxonomy_dir,
            total_samples,
            accessions_file,
            taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            dataset_distribution=dataset_distribution,
            genome_cache_size=genome_cache_size)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers, drop_last)

    def enable_multithreading_if_possible(self):
        if self.dataset.genome_cache_is_full():
            try:
                if self.num_workers != self._num_workers:
                    self.num_workers = self._num_workers
                    self.logger.info(f'Enabling {self.num_workers} '
                                     'workers for data loading...')
            except AttributeError:
                pass
        else:
            self._num_workers = self.num_workers
            self.num_workers = 0

    def step(self, epoch):
        super().step(epoch)
        self.enable_multithreading_if_possible()
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            seed = epoch
            seed = seed * get_world_size() + get_global_rank()
            if self.valid_loader:
                seed = 2**32 - seed
                self.dataset.set_to_lognorm()
            else:
                self.dataset.set_to_default_sim()
            self.dataset.simulator.set_seed(seed)

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True


class RefSeqProkaryotaKmerBagsDataLoader(BaseDataLoader):
    """
    RefSeq DataLoader

    Any encoding or transformation should be done here
    and passed to the Dataset
    """
    def __init__(self,
                 target_format,
                 genome_dir,
                 taxonomy_dir,
                 total_bags,
                 bag_size,
                 batch_size,
                 kmer_vocab_file,
                 fixed_dataset=False,
                 drop_last=False,
                 dataset_distribution='uniform',
                 reseed_every_n_bags=1,
                 accessions_file=None,
                 taxids_list=None,
                 error_model=None,
                 rmin=None,
                 rmax=None,
                 download=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 noise=None,
                 forward_reads_only=False,
                 filter_by_level=None,
                 num_to_keep=1,
                 genome_cache_size=1000,
                 single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        ncbi_email = 'your-email@domain.com'
        ncbi_api = None

        g2read = mytransforms.GenomeToNoisyKmerRead(
            kmer_vocab_file,
            error_model,
            rmin,
            rmax,
            p=noise,
            forward_reads_only=forward_reads_only)

        self.rmin = rmin
        self.rmax = rmax
        self.noise = noise
        self.error_model = error_model
        self.fixed_dataset = fixed_dataset

        self.vocab_size = len(g2read.kmer2id)
        if self.error_model:
            self.rmin = g2read.rmin
            self.rmax = g2read.rmax

        trsfm_x = mytransforms.Compose([
            g2read,
            mytransforms.ToTensorWithView(dtype=torch.long, view=[-1])
        ])
        trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])

        self.valid_loader = False

        self.dataset = datasets.RefSeqProkaryotaBags(
            bag_size=bag_size,
            total_bags=total_bags,
            target_format=target_format,
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=trsfm_x,
            transform_y=trsfm_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            dataset_distribution=dataset_distribution,
            reseed_every_n_bags=reseed_every_n_bags,
            genome_cache_size=genome_cache_size,
            num_workers=num_workers,
            single_read_target_vectors=single_read_target_vectors)

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         0, drop_last)

    def step(self, epoch):
        super().step(epoch)
        if not self.fixed_dataset:
            self.dataset.idx_offset = epoch * len(self.dataset)
            if self.valid_loader:
                self.dataset.set_to_lognorm()
            else:
                self.dataset.set_to_default_sim()

    def init_validation(self, other):
        super().init_validation(other)
        self.fixed_dataset = other.fixed_dataset
        self.valid_loader = True
