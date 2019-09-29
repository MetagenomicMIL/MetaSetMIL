import numpy as np
from utils.genome_utils import GenomeTools
import matplotlib.pyplot as plt
import math


class MetagenomeSimulator(object):
    def get_accession_index(self, i):
        raise NotImplementedError

    def set_seed(self, seed):
        raise NotImplementedError


class UniformSimulator(MetagenomeSimulator):
    def __init__(self, accessions):
        self.random = np.random.RandomState(seed=0)
        self.accessions = accessions

    def get_accession_index(self, idx):
        self.random.seed(idx)
        id_idx = self.random.randint(low=0, high=len(self.accessions))
        return id_idx

    def set_seed(self, seed):
        pass


class UniformAtLevelSimulator(MetagenomeSimulator):
    def __init__(self,
                 genome_paths,
                 acc2taxid,
                 taxtree,
                 tax_categories,
                 start_level='genus'):
        accessions = [g.stem for g in genome_paths]
        self.acc2index = {g: i for i, g in enumerate(accessions)}
        self.tax_categories = tax_categories
        self.start_level = start_level

        taxtree.trim_tree([acc2taxid[ac] for ac in accessions])

        taxid2acc = {}
        for ac in accessions:
            try:
                taxid2acc[acc2taxid[ac]].append(ac)
            except KeyError:
                taxid2acc[acc2taxid[ac]] = [ac]

        self.start_ids = taxtree.get_ids_at_level(self.start_level)
        self.leveltax2acc = {}

        for id in self.start_ids:
            accessions_for_id = []
            leaf_ids = taxtree.get_all_leaves(id)
            for lid in leaf_ids:
                accessions_for_id.extend(taxid2acc[lid])
            self.leveltax2acc[id] = accessions_for_id

        self.random = np.random.RandomState(seed=0)

    def get_accession_index(self, idx):
        self.random.seed(idx)
        taxid = self.random.choice(self.start_ids)
        accessions = self.leveltax2acc[taxid]
        acc = self.random.choice(accessions)
        id_idx = self.acc2index[acc]
        return id_idx

    def set_seed(self, seed):
        pass


class LogNormalSimulator(MetagenomeSimulator):
    def __init__(self,
                 genome_paths,
                 acc2taxid,
                 taxtree,
                 tax_categories,
                 start_level='genus',
                 max_strains_per_id=10,
                 use_genome_lengths=False):
        self.genome_paths = genome_paths
        self.genome_lengths = 1
        if use_genome_lengths:
            self.genome_lengths = np.array(
                [len(GenomeTools.read_fasta(g)) for g in genome_paths])

        self.accessions = [g.stem for g in self.genome_paths]
        self.acc2index = {g: i for i, g in enumerate(self.accessions)}
        self.acc2taxid = acc2taxid

        self.tax_categories = tax_categories
        self.start_level = start_level
        self.max_strains_per_id = max_strains_per_id
        self.seed = 1

        self.taxtree = taxtree

        taxid2acc = {}
        for ac in self.accessions:
            try:
                taxid2acc[self.acc2taxid[ac]].append(ac)
            except KeyError:
                taxid2acc[self.acc2taxid[ac]] = [ac]

        leveltaxid2acc = {}
        start_ids = taxtree.get_ids_at_level(self.start_level)
        for id in start_ids:
            accs_for_id = []
            leaf_ids = taxtree.get_all_leaves(id)
            for lid in leaf_ids:
                accs_for_id.extend(taxid2acc[lid])
            leveltaxid2acc[id] = accs_for_id

        unmapped_accs = list(
            set(self.accessions) -
            set([id for l in leveltaxid2acc.values() for id in l]))
        if unmapped_accs:
            print('WARNING: Some accessions are not mapped to the tree')
            print(unmapped_accs)

        self.start_ids = start_ids
        self.leveltaxid2acc = leveltaxid2acc

        self.random = np.random.RandomState(seed=0)
        self.random_main = np.random.RandomState(seed=self.seed)

        self.probs = None
        self.abundances = np.zeros(len(self.accessions))

        self.new_community()

    def new_community(self):
        # print('Generating new microbial community...')
        self.abundances.fill(0)
        self.random_main.seed(self.seed)

        start_ids = self.start_ids
        start_ab = self.random_main.lognormal(mean=1.0,
                                              sigma=2.0,
                                              size=len(start_ids))
        for id, ab in zip(start_ids, start_ab):
            accessions = self.leveltaxid2acc[id]

            # accessions holds all genomes assigned to that id
            num_of_genomes = self.random_main.geometric(
                2.0 / self.max_strains_per_id)
            num_of_genomes = min(num_of_genomes, len(accessions))

            selected = self.random_main.choice(len(accessions),
                                               size=num_of_genomes,
                                               replace=False)
            Y = self.random_main.lognormal(mean=1.0,
                                           sigma=2.0,
                                           size=num_of_genomes)
            Ysum = np.sum(Y)
            for g_index, y_i in zip(selected, Y):
                acc = accessions[g_index]
                ab_i = (y_i / Ysum) * ab
                index = self.acc2index[acc]
                self.abundances[index] = ab_i

        ab_times_lengths = self.abundances * self.genome_lengths
        self.probs = ab_times_lengths / np.sum(ab_times_lengths)

    def create_plots(self):
        abundances_per_level = [{} for _ in self.tax_categories]
        for acc, ab in zip(self.accessions, self.abundances):
            id = self.acc2taxid[acc]
            taxline = self.taxtree.collect_path(id, self.tax_categories)
            for t, ab_dict in zip(taxline, abundances_per_level):
                try:
                    ab_dict[t] += ab
                except KeyError:
                    ab_dict[t] = ab

        # Whittaker plots per level
        columns = math.ceil(len(self.tax_categories) / 2)
        f, axarr = plt.subplots(2, columns)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for index, (level, abs_dict) in enumerate(
                zip(self.tax_categories, abundances_per_level)):
            xindex = index % 2
            yindex = index // 2
            abs = list(abs_dict.values())
            abs = np.array(abs)
            abs = abs / np.sum(abs) * 100
            abs.sort()
            abs[abs == 0] = np.nan
            abs = np.flip(abs)
            ax = axarr[xindex, yindex]
            ax.plot(np.arange(len(abs)),
                    abs,
                    color='red',
                    marker='_',
                    linewidth=2.0,
                    markersize=10)
            ax.set_xlim(right=len(abs) + 1)
            ax.grid(axis='y')
            ax.set_yscale('log')
            ax.set_xlabel('Rank')
            ax.set_ylabel('% relative abundance')
            ax.set_title(level)
        f.suptitle('Rank abundance curve', fontsize=16)
        plt.show()

    def get_accession_index(self, idx):
        self.random.seed(idx)
        id_idx = self.random.choice(len(self.accessions), p=self.probs)
        return id_idx

    def set_seed(self, seed):
        self.seed = seed
        self.new_community()
        # self.create_plots()
