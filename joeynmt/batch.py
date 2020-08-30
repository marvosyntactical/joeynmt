# coding: utf-8

"""
Implementation of a mini-batch.
"""
from joeynmt.data import TorchBatchWithKB


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).sum().item() # target tokens

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index



class Batch_with_KB(Batch):
    """Object for holding a batch of data with mask and with a knowledgebase during training.
    Input is a batch from a torch text iterator created by 
    joeynmt.data.make_data_iter_batch_size_1 .
    """

    def __init__(self, TBatchWithKB: TorchBatchWithKB, pad_index, use_cuda=False):
        """
        Create a new joey batch from a TBatchWithKB.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param TBatchWithKB:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = TBatchWithKB.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda
        #knowledgebase:
        self.kbsrc = TBatchWithKB.kbsrc[0] # .kbsrc, .kbtrg are tuples of lines, lengths # TODO do i need lengths?
        self.kbtrg = TBatchWithKB.kbtrg[0]
        self.kbtrv = TBatchWithKB.kbtrv #not indexed because include_lengths is false for trv field  #FIXME empty

        assert self.kbsrc.shape[0] == self.kbtrg.shape[0] == self.kbtrv.shape[0], "batch dimensions should be the same"

        if hasattr(TBatchWithKB, "trg"):
            trg, trg_lengths = TBatchWithKB.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).sum().item()

        if hasattr(TBatchWithKB, "trgcanon"):
            # these attributes are checked for in model.get_loss_for_batch if torch no grad 
            # to report validation loss on canonized target data and still be able to compute bleu on raw target data
            trgcanon, trgcanon_lengths = TBatchWithKB.trgcanon
            self.trgcanon_input = trgcanon[:, :-1]
            self.trgcanon_lengths = trgcanon_lengths
            self.trgcanon = trgcanon[:, 1:]
            self.trgcanon_mask = (self.trgcanon_input != pad_index).unsqueeze(1)
            self.ntokenscanon = (self.trgcanon != pad_index).sum().item()

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

        if hasattr(self, "trgcanon"):
            self.trgcanon_input = self.trgcanon_input.cuda()
            self.trgcanon = self.trgcanon.cuda()
            self.trgcanon_mask = self.trgcanon_mask.cuda()
        
        # move kb to cuda, too!
        self.kbsrc = self.kbsrc.cuda()
        # not sure if this is only sensible for batches that
        # get embedded at some point later on but here goes:
        self.kbtrg = self.kbtrg.cuda() # FIXME TODO probably actually dont move these two to cuda??
        self.kbtrv = self.kbtrv.cuda()


    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src = self.src[perm_index]
        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src_mask = self.src_mask[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask


        if self.trg_input is not None:
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]
        
            self.trg_lengths = sorted_trg_lengths
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg = sorted_trg

        if hasattr(self, trgcanon_input):
            sorted_trgcanon_lengths = self.trgcanon_lengths[perm_index]
            sorted_trgcanon_input = self.trgcanon_input[perm_index]
            sorted_trgcanon_mask = self.trgcanon_mask[perm_index]
            sorted_trgcanon = self.trgcanon[perm_index]
        
            self.trg_lengths = sorted_trgcanon_lengths
            self.trgcanon_input = sorted_trgcanon_input
            self.trgcanon_mask = sorted_trgcanon_mask
            self.trgcanon = sorted_trgcanon

        if self.use_cuda:
            self._make_cuda()

        return rev_index
