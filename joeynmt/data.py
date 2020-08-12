# coding: utf-8
"""
Data module
"""
import sys
import random
import io
import os
import os.path
from typing import Optional, List
from copy import deepcopy
from pprint import pprint

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, Batch

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

    
def pkt_tokenize(s)-> List:
    s = s+" "
    pkt = ".,?!-;:()" # NOTE candidates: '
    space = ["\t", "\n", " "]

    r = []
    split = list(s)
    curr = []
    for c in split:
        if c in space:
            token = "".join(curr)
            if token: 
                r += [token]
            curr = []
        else:
            if c in pkt:
                token = "".join(curr)  
                if token: 
                    r += [token]
                curr = []
            
            curr += [c] # add pkt to tokens, but not whitespace
    return r

def tokenize(s):
    return s.split()

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
    max_sent_length = data_cfg.get("max_sent_length", sys.maxsize*.1)

    #kb stuff
    kb_task = bool(data_cfg.get("kb_task", False))

    if kb_task:
        kb_src = data_cfg.get("kb_src", "kbk")
        kb_trg = data_cfg.get("kb_trg", "kbv")
        kb_lkp = data_cfg.get("kb_lkp", "lkp")
        kb_len = data_cfg.get("kb_len", "len")
        kb_trv = data_cfg.get("kb_truvals", "trv")
        global_trv = data_cfg.get("global_trv", "global.trv")
        trutrg = data_cfg.get("trutrg", "car") 

        # TODO: following is hardcoded; add to configs please
        pnctprepro = True
    else: pnctprepro = False
    # default joeyNMT behaviour for sentences

    tok_fun = list if level == "char" else (pkt_tokenize if pnctprepro else tokenize)

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

    if kb_task:
        # NOTE lowercase MUST be False for datasets with tokens that may include whitespace!
        # the torchtext lowercase pipeline seems to operate not just on the first char of a token (dataset field level)
        # but lowercases individual words separated by whitespace WITHIN a specified token
        # which leads to the vocab not recognizing tokens even though added to the field.vocab
        # via joeynmt.vocabulary.build_vocab
        # other way to circumvent may be to lowercase in the same manner before calling
        # field.process
        trv_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                            pad_token=PAD_TOKEN, tokenize=lambda entire_line: [entire_line],
                            unk_token=UNK_TOKEN,
                            batch_first=True, lower=False,
                            include_lengths=False)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred =
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)
    
       
    if kb_task: #load train_kb and metadata

        # NOTE change trg_lang to trutrg for dev/test 
        # train_data has been loaded with normal extension (canonized files, e.g. train.carnon)
        # dev/test_data will be loaded from non canonized files
        trg_lang = trutrg

        train_kb_truvals = MonoDataset(path=train_path,
                                    ext=("."+kb_trv),
                                    field=("kbtrv", trv_field),
                                    filter_pred= lambda x:True
                                    )

        train_kb = TranslationDataset(path=train_path,
                                    exts=("." + kb_src, "." + kb_trg),
                                    fields=(("kbsrc", src_field), ("kbtrg", trg_field)),
                                    filter_pred=
                                    lambda x: True)
                                   
        with open(train_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        train_kb_lookup = [int(elem[:-1]) for elem in lookup if elem[:-1]]
        with open(train_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        train_kb_lengths = [int(elem[:-1]) for elem in lengths if elem[:-1]]
            

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)

    trg_vocab_file = data_cfg.get("trg_vocab", None)
    trg_kb_vocab_file = data_cfg.get("trg_kb_vocab", None)
    trg_vocab_file = trg_vocab_file if not trg_kb_vocab_file else trg_kb_vocab_file #prefer to use joint trg_kb_vocab_file is specified

    vocab_building_datasets = train_data if not kb_task else (train_data, train_kb)
    vocab_building_src_fields = "src" if not kb_task else ("src", "kbsrc")
    vocab_building_trg_fields = "trg" if not kb_task else ("trg", "kbtrg")

    pkld_src_voc = data_cfg.get("src_voc_pkl", "data/voc/src.p")
    pkld_trg_voc = data_cfg.get("trg_voc_pkl", "data/voc/trg.p")
    

    # TODO figure out how to serialize/pickle Vocabulary objects

    src_vocab = build_vocab(fields=vocab_building_src_fields, min_freq=src_min_freq, max_size=src_max_size, dataset=vocab_building_datasets, vocab_file=src_vocab_file)

    trg_vocab = build_vocab(fields=vocab_building_trg_fields, min_freq=trg_min_freq,max_size=trg_max_size, dataset=vocab_building_datasets, vocab_file=trg_vocab_file)

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
        dev_kb = TranslationDataset(path=dev_path,
                                exts=("." + kb_src, "." + kb_trg),
                                fields=(("kbsrc",src_field), ("kbtrg",trg_field)),
                                filter_pred=
                                lambda x: True)
        dev_kb_truvals = MonoDataset(path=dev_path,
                                ext=("."+kb_trv),
                                field=("kbtrv", trv_field),
                                filter_pred= lambda x:True
                                )
   
        with open(dev_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        dev_kb_lookup = [int(elem[:-1]) for elem in lookup if elem[:-1]]
        with open(dev_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        dev_kb_lengths = [int(elem[:-1]) for elem in lengths if elem[:-1]]
            
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
        test_kb = TranslationDataset(path=test_path,
                                exts=("." + kb_src, "." + kb_trg),
                                fields=(("kbsrc", src_field), ("kbtrg", trg_field)),
                                filter_pred=
                                lambda x: True)
        test_kb_truvals = MonoDataset(path=test_path,
                                ext=("."+kb_trv),
                                field=("kbtrv", trv_field),
                                filter_pred= lambda x:True
                                )
  
        with open(test_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        test_kb_lookup = [int(elem[:-1]) for elem in lookup if elem[:-1]]
        with open(test_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        test_kb_lengths = [int(elem[:-1]) for elem in lengths if elem[:-1]]

    # finally actually set the .vocab field attributes        
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    if kb_task:
        # NOTE this vocab is hardcodedly built from the concatenation of train+dev+test trv files!
        trv_path = train_path[:len(train_path)-train_path[::-1].find("/")]+global_trv

        assert os.path.isfile(trv_path)

        trv_vocab = deepcopy(trg_vocab)
        trv_vocab._from_file(trv_path)
        
        assert trg_vocab.itos == trv_vocab.itos[:len(trg_vocab)]

        print(f"Added true value lines as tokens to trv_vocab of length={len(trv_vocab)}")
        trv_field.vocab = trv_vocab



    if not kb_task: #default values for normal pipeline
        train_kb, dev_kb, test_kb = None, None, None
        trv_vocab = None
        train_kb_lookup, dev_kb_lookup, test_kb_lookup = [],[],[]
        train_kb_lengths, dev_kb_lengths, test_kb_lengths = [],[],[]
        train_kb_truvals, dev_kb_truvals, test_kb_truvals = [],[],[]
    

    return train_data, dev_data, test_data,\
        src_vocab, trg_vocab,\
        train_kb, dev_kb, test_kb,\
        train_kb_lookup, dev_kb_lookup, test_kb_lookup,\
        train_kb_lengths, dev_kb_lengths, test_kb_lengths,\
        train_kb_truvals, dev_kb_truvals, test_kb_truvals,\
        trv_vocab


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
    def __init__(self, data=None, dataset=None, kb_data=None, kb_truval_data=None, device=None):
        
        """
        Create a TorchBatchWithKB from a list of examples.
        Cant be put in .batch because of variable name clash
        between torchtext batch and joeynmt batch :(
        """

        if data is not None:
            assert hasattr(data, "kb") 
            self.batch_size = len(data)
            self.dataset = dataset
            self.kb_data = kb_data
            self.kb_truval_data = kb_truval_data

            joint_field_dict = deepcopy(dataset.fields)
            joint_field_dict.update(kb_data.fields)

            self.fields = joint_field_dict.keys()  # copy field names

            self.input_fields = [k for k, v in dataset.fields.items() if
                                    v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                    v is not None and v.is_target]
            self.knowledgebase_fields = [k for k,v in kb_data.fields.items() if v is not None]

            for (name, field) in self.dataset.fields.items():
                print(f"processing field: {(name, field)}")
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    print(f"batch.{name} unprocessed={batch}")
                    setattr(self, name, field.process(batch, device=device))
                    #print(f"self.{name}={getattr(self, name)}")
            #print(f"data fields: {self.dataset.fields.items()}")
            
            for (name, field) in self.kb_truval_data.fields.items():
                if field is not None:
                    truvals = [["<s>"]]
                    truvals += [getattr(x, name) for x in data.kbtrv]
                    print(f"batch.trv unprocessed={truvals}")
                    setattr(self, name, field.process(truvals, device=device))
                    print(f"self.{name}={getattr(self, name)}")
            """ TODO: remove this 
            print(f"kbtrv fields: {self.kb_truval_data.fields.items()}")
            print(f"kbtrv field vocab: {self.kb_truval_data.fields['kbtrv'].vocab.stoi}")
            print(f"kbtrv vocab.itos[7]: {self.kb_truval_data.fields['kbtrv'].vocab.itos[7]}")
            print(f"kbtrv vocab.stoi['550 Alester Ave']: {self.kb_truval_data.fields['kbtrv'].vocab.stoi['550 Alester Ave']}")
            print(f"kbtrv vocab.stoi['parking garage']: {self.kb_truval_data.fields['kbtrv'].vocab.stoi['parking garage']}")
            print(f"kbtrv field batch sentences: {self.kb_truval_data.fields['kbtrv'].vocab.arrays_to_sentences(self.kbtrv)}")
            """

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
    adds kb fields to minibatch list in KB_Iterator
    """
    def __init__(self, *args, **kwargs):
        super(KB_minibatch, self).__init__(*args, **kwargs)
        self.kb = None
        self.kbtrv = None


def batch_with_kb(data, kb_data, kb_lkp, kb_lens, kb_truvals):
    print(len(kb_data), len(kb_truvals))

    # minibatch.kb length, adds it to this attribute and yields
    # elements from data in chunks of conversations

    minibatch = KB_minibatch()
    current = 0
    corresponding_kb = 0
    kb_len = 0

    for i, ex in enumerate(data):
        #print(f"batch_with_kb: loop begin current index in train.kb(.*): {current}")

        last_corresponding_kb = corresponding_kb
        corresponding_kb = kb_lkp[i]

        if corresponding_kb != last_corresponding_kb:

            yield minibatch
            minibatch = KB_minibatch()
            #print(f"batch_with_kb: adding {kb_len} to current {current}")
            current += kb_len
            
        kb_len = kb_lens[corresponding_kb]

        minibatch.kb = kb_data[current:current+kb_len]
        minibatch.kbtrv = kb_truvals[current:current+kb_len]

        assert len(minibatch.kb) == len(minibatch.kbtrv), (len(minibatch.kb),len(minibatch.kbtrv)) 

        minibatch.append(ex)

        # debug start:
        previous_kb_len = kb_lens[last_corresponding_kb] 
        previous_kb = kb_data[current-previous_kb_len:current]
        previous_trv = kb_truvals[current-previous_kb_len:current]

        print()
        print(f"minibatch.kb length: {len(minibatch.kb)}")
        print(f"minibatch.kb:")
        pprint([(entry.kbsrc, entry.kbtrg, tru.kbtrv) for entry, tru in zip(minibatch.kb,minibatch.kbtrv)], width=110)
        print()
        print(f"minibatch.src/trg: {(ex.src, ex.trg)}")
        print()
        print("batch_with_kb: current, kb_len, current+kb_len: ",current, kb_len,current+kb_len)
        print()
        print(f"corresponding_kb: {corresponding_kb}")
        """ # to see if previous kb is a match:
        print(f"previous minibatch was this long: {len(previous_kb)}")
        print(f"previous minibatch should have been:")
        pprint([(entry.kbsrc, entry.kbtrg, tru.kbtrv) for entry, tru in zip(previous_kb,previous_trv)], width=110)
        print()
        """

        # assert i+1 < 50, "<^ check out this line in data/kvr/{train|dev}.lkp, which says e.g. 24, then do 'head -n 24 {train|dev}.len | awk '{s+=$1} END {print s}'"
        """
        FIXME: there seems to be an overlap between some knowledgebases, e.g. in train.lkp:
        v CORRECT, PIZZA CHICAGO BELONGS HERE v
        line 54 -> kb #22 -> train.kb[756:756+32=788] (32 is line #22+1 in train.len; 756 result of above awk command (sum lines 0-22))
        v WRONG, PIZZA CHICAGO DOESNT BELONG HERE v
        line 55 -> kb #23 -> train.kb[784:784+28=812] (28 is line #23+1 in train.len; 788 result of above awk command!!!!!!) <- FIXME NOTE TODO

        NOTE:
        Fixed problem: moved kb_len assignment after minibatch yield; now taking the length of the current knowledgebase, not the length of the next one...

        FIXME:
        New problem: mismatch in dev data, e.g. i+1<400 => dev.lkp[400] is 159, awk cmd with 159
        returns 4917, but variable 'current' is at 5339...
        also: the kb batch from 5339 to 5339+kb_len does not even align with any kb ... its off by 1 at least
        => run modele to do validation only 
        """
        # debug end

    if minibatch:
        yield minibatch


class KB_Iterator(Iterator):
    """
    Iterates over a kb in parallel to the dataset
    individual batches should have an attribute .kb
    which is equal to the corresponding kb Tensor (m x 3)

    for the moment,
    sorting and shuffling is ignored and not done
    """

    def __init__(self, dataset, kb_dataset, kb_lkp, kb_lens, kb_truvals, sort_key=None, device=None,\
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
        self.kb_truvals = kb_truvals
        
    def create_batches(self):
        self.batches = batch_with_kb(self.data(), self.kb_data, self.kb_lkp, self.kb_lens,self.kb_truvals)

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
                batch = TorchBatchWithKB(minibatch, self.dataset, self.kb_data, self.kb_truvals, self.device)
                assert hasattr(batch, "kbsrc"), dir(batch)
                assert hasattr(batch, "kbtrg"), dir(batch)
                yield batch
            if not self.repeat:
                return

       
class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex, field_name="src"): #no idea how to access this or when it is used TODO find that out!
        return len(eval(f"ex.{field_name}"))

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the field that will be used for data. Can also be tuple of (name, field)
        :param kwargs: Passed to the constructor of data.Dataset.
        """
        fields = [('src', field)] if not type(field) == type((0,0)) else [field]

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
                    kb_truvals: MonoDataset,
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


    if train: # TODO remove this option entirely?
        # optionally shuffle and sort during training
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens, kb_truvals,\
            repeat=False, sort=False, 
            train=True, shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens, kb_truvals,\
            repeat=False, train=False, sort=False, sort_within_batch=True)

    return data_iter
