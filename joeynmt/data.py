# coding: utf-8
"""
Data module
"""
import sys
import random
import io
import os
import os.path
import pickle
from typing import Optional
from copy import deepcopy

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, Batch

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    use this function instead of load_data when data_cfg["kb_task"] == True

    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
        - train_kb: TranslationDataset from train KB
        - dev_kb: TranslationDataset from dev KB
        - test_kb: TranslationDataset from test KB
        - train_kb_lookup: List of KB association lookup indices from data to KBs
        - dev_kb_lookup: List of KB association lookup indices from data to KBs
        - test_kb_lookup: List of KB association lookup indices from data to KBs
        - train_kb_lengths: List of KB lengths
        - dev_kb_lengths: List of KB lengths
        - test_kb_lengths: List of KB lengths
        

    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    #kb stuff
    kb_task = bool(data_cfg.get("kb_task", False))
    if kb_task:
        kb_src = data_cfg.get("kb_src", "kbk")
        kb_trg = data_cfg.get("kb_trg", "kbv")
        kb_lkp = data_cfg.get("kb_lkp", "lkp")
        kb_len = data_cfg.get("kb_len", "len")

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    #kb stuff
    if kb_task:
        assert not (level=="char") #TODO take out/add char compatibility completely
        kb_tok_fun = lambda s: s.split("::")


    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)
    
    if kb_task: #load train_kb and metadata
        train_kb = TranslationDataset(path=train_path,
                                    exts=("." + kb_src, "." + kb_trg),
                                    fields=(("kbsrc",src_field), ("kbtrg",trg_field)),
                                    filter_pred=
                                    lambda x: True)
                                   
        with open(train_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        train_kb_lookup = [int(elem[:-1]) for elem in lookup]
        with open(train_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        train_kb_lengths = [int(elem[:-1]) for elem in lengths]


    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)

    trg_vocab_file = data_cfg.get("trg_vocab", None)
    trg_kb_vocab_file = data_cfg.get("trg_kb_vocab", None)
    trg_vocab_file = trg_vocab_file if not trg_kb_vocab_file else trg_kb_vocab_file


    pkld_src_voc = data_cfg.get("src_voc_pkl", "data/voc/src.p")
    pkld_trg_voc = data_cfg.get("trg_voc_pkl", "data/voc/trg.p")
    
    prevent_pkl_load_voc = data_cfg.get("prevent_pkl_load_voc", None)

    if not prevent_pkl_load_voc and os.path.isfile(pkld_src_voc):
        with open(pkld_src_voc, "rb") as filehandler:
            src_vocab = pickle.load(filehandler)
    else:
        src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                                max_size=src_max_size,
                                dataset=train_data, vocab_file=src_vocab_file)
        if not os.path.isfile(pkld_src_voc):
            with open(pkld_src_voc, "wb") as filehandler:
                try:
                    pickle.dump(src_vocab, filehandler)
                except Exception as e:
                    os.remove(pkld_src_voc)
                    print(e)
                    print(e.with_traceback)

    if not prevent_pkl_load_voc and os.path.isfile(pkld_trg_voc):
        with open(pkld_trg_voc, "rb") as filehandler:
            trg_vocab = pickle.load(filehandler)
    else:
        trg_vocab = build_vocab(field="src", min_freq=src_min_freq,
                                max_size=src_max_size,
                                dataset=train_data, vocab_file=trg_vocab_file)

        if not os.path.isfile(pkld_trg_voc):
            with open(pkld_trg_voc, "wb") as filehandler:
                try:
                    pickle.dump(trg_vocab, filehandler)
                except Exception as e:
                    os.remove(pkld_trg_voc)
                    print(e)
                    print(e.with_traceback)


    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)


    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep

    
    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))

    if kb_task: #load dev kb and metadata
        dev_kb = TranslationDataset(path=train_path,
                                exts=("." + kb_src, "." + kb_trg),
                                fields=(("kbsrc",src_field), ("kbtrg",trg_field)),
                                filter_pred=
                                lambda x: True)
    
        with open(dev_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        dev_kb_lookup = [int(elem[:-1]) for elem in lookup]
        with open(dev_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        dev_kb_lengths = [int(elem[:-1]) for elem in lengths]

    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    if kb_task: #load test kb and metadata
        test_kb = TranslationDataset(path=train_path,
                                exts=("." + kb_src, "." + kb_trg),
                                fields=(("kbsrc",src_field), ("kbtrg",trg_field)),
                                filter_pred=
                                lambda x: True)

        with open(test_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        test_kb_lookup = [int(elem[:-1]) for elem in lookup]
        with open(dev_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        test_kb_lengths = [int(elem[:-1]) for elem in lengths]

    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    if not kb_task: #default values for normal pipeline
        train_kb, dev_kb, test_kb = None, None, None
        train_kb_lookup, dev_kb_lookup, test_kb_lookup = [],[],[]
        train_kb_lengths, dev_kb_lengths, dev_kb_lengths = [],[],[]


    return train_data, dev_data, test_data,\
        src_vocab, trg_vocab,\
        train_kb, dev_kb, test_kb,\
        train_kb_lookup, dev_kb_lookup, test_kb_lookup,\
        train_kb_lengths, dev_kb_lengths, dev_kb_lengths



# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class TorchBatchWithKB(Batch):
    #inherits from torch batch, not joey batch!
    def __init__(self, data=None, dataset=None, kb_data=None, device=None):
        """Create a TorchBatchWithKB from a list of examples.
        Cant be put in .batch because of variable name clash
        between torchtext batch and joeynmt batch :(
        """

        if data is not None:
            assert hasattr(data, "kb") 
            self.batch_size = len(data)
            self.dataset = dataset
            self.kb_data = kb_data
            joint_field_dict = deepcopy(dataset.fields)
            joint_field_dict.update(kb_data.fields)
            self.fields = joint_field_dict.keys()  # copy field names

            self.input_fields = [k for k, v in dataset.fields.items() if
                                    v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                    v is not None and v.is_target]
            self.knowledgebase_fields = [k for k,v in kb_data.fields.items() if v is not None]

            for (name, field) in self.dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))
            for (name, field) in self.kb_data.fields.items():
                if field is not None:
                    kb = [] # used to be KB_minibatch
                    if name == "kbsrc":
                        kb.append(["<s>"]) #dummy elem key
                    elif name == "kbtrg":
                        kb.append(["<s>"]) #dummy elem val
                    
                    # dummy kb entry for scheduling task, added to all kbs as first entry
                    # TODO what should dummy tokens be? shouldnt affect default decoder behavior..
                    # the kb has only the dummy elem iff theres null items in the scenario
                    # which happens iff 
                    # the task is of calendar scheduling type
                    # AND
                    # the driver is only tasked with making an appointment, not requesting one
                    # (NOTE current implementation) add dummy element in zeroth place every single time

                    kb += [getattr(x,name) for x in data.kb]
                    if len(kb) > 1:
                        print(len(kb), type(kb), [type(x) for x in kb])
                        print(len(data.kb), type(data.kb)) #data.kb list of examples
                        print(data.kb[0])
                        print(type(data.kb[0]))
                        print(getattr(data.kb[0],name))

                    setattr(self, name, field.process(kb, device=device))

                    print(f"batch.{name}: ", eval(f"self.{name}"),type(eval(f"self.{name}")))

                    # TODO: find out what the second batch.kbsrc/kbtrg tensor is!
                    # we discard it anyways in model.get_loss_for_batch


                else:
                    raise ValueError(kb_data.field)

    @classmethod
    def fromvars(cls, dataset, kb_data, batch_size, train=None, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.kb_data = kb_data
        batch.fields = dataset.fields.keys()
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch


class KB_minibatch(list):
    """
    adds kb field to minibatch list in KB_Iterator
    """
    def __init__(self, *args, **kwargs):
        super(KB_minibatch, self).__init__(*args, **kwargs)
        self.kb = None


def batch_with_kb(data, kb_data, kb_lkp, kb_lens):
    # for KB_Iterator. uses kb_lkp and kb_lens to determine 
    # minibatch.kb length, adds it to this attribute and yields
    # elements from data in chunks of conversations
    # TODO: try holding the entire prior conversation as source

    
    minibatch = KB_minibatch()
    current = 0
    corresponding_kb = 0
    for i, ex in enumerate(data):
        last_kb = corresponding_kb
        corresponding_kb = kb_lkp[i]
        kb_len = kb_lens[corresponding_kb]

        if kb_lkp[i] != last_kb:
            yield minibatch
            minibatch = KB_minibatch()
            current += kb_len

        minibatch.kb = kb_data[current:current+kb_len]
        minibatch.append(ex)


    if minibatch:
        yield minibatch


class KB_Iterator(Iterator):
    """
    Iterates over a kb in parallel to the dataset
    individual batches should have an attribute .kb
    which is equal to the corresponding kb Tensor (m x 3)

    for the moment,
    sorting and shuffling is ignored and not done

    TODO additional params:

    """

    def __init__(self, dataset, kb_dataset, kb_lkp, kb_lens, sort_key=None, device=None,\
    batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=None):
        """
        takes additional args:
        :param kb_data: monodataset created in load_data from kb 
        :param kb_lkp: list of association lookups between dataset examples and \
            kbs in kb_data
        :param kb_len: list of kb lengths within kb_data

        
        :return data_iter: data iterator object where each batch
        has an associated knowledgebase with it 
        """
        batch_size=1 #TODO remove batch_size specification /look at Iterator implementation of init to see if this does anything
        
        super(KB_Iterator, self).__init__(dataset, batch_size, sort_key=None, device=None,\
    batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=None)
        self.kb_data = kb_dataset
        self.kb_lkp = kb_lkp
        self.kb_lens= kb_lens
        
    def create_batches(self):
        self.batches = batch_with_kb(self.data(),self.kb_data, self.kb_lkp, self.kb_lens)

    def data(self):
        return self.dataset
    
    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = TorchBatchWithKB(minibatch, self.dataset, self.kb_data, self.device)
                assert hasattr(batch, "kbsrc"), dir(batch)
                assert hasattr(batch, "kbtrg"), dir(batch)
                yield batch
            if not self.repeat:
                return

       
class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)



def make_data_iter_kb(dataset: Dataset,
                    kb_data: TranslationDataset,
                    kb_lkp: list,
                    kb_lens: list,
                    batch_size: int,
                    batch_type: str = "sentence",
                    train: bool = False,
                    shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext Iterator (not BucketIterator) for a torchtext dataset.
    Uses batch_size = 1
    TODO make batch size variable based on kb length

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """


    if train:
        # optionally shuffle and sort during training
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens,\
            repeat=False, sort=False, 
            train=True, shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens,\
            repeat=False, train=False, sort=False, sort_within_batch=True)

    return data_iter
