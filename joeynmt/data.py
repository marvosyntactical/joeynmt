# coding: utf-8
"""
Data module
"""
import sys
import random
import io
import os
import os.path
from typing import Optional, List, Tuple
from copy import deepcopy
from pprint import pprint

import numpy as np


from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, Batch

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

# FIXME this import from scripts needed to canonize scheduling requests to fill empty scheduling KBs
from data.scripts_kvr.canonize import load_json, preprocess_entity_dict, canonize_sequence
# joeynmt.data.canonize_sequence is referred to form other parts of joeynmt (.metrics; maybe .helpers) FIXME

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

def hash_canons(tokzd_sequence: List[str], vocab_token_list: List[str]) -> (List[str], List[int], List[Tuple[str, List[str]]]):
    """
    To canonize dstc2; mapping vocab to canonical is 1:1

    :param tokzd_sequence:
    :param vocab_token_list:
    :return processed: tokzd sequence with canonical tokens hashed 
    :return indices: list(range(len(tokzd_sequence)))
    :return matches: list of tuple: (hash_val: [orig token])
    """
    processed = [str(hash(tok)) if tok in vocab_token_list else tok for tok in tokzd_sequence]
    indices = list(range(len(tokzd_sequence)))
    matches = []
    for i, tok in enumerate(processed):
        raw_curr = tokzd_sequence[i]
        if str(hash(raw_curr)) == tok:
            # hashed tok 
            lkp = [] # indices that were also mapped to this token
            for idx in range(len(processed)):
                raw_other = tokzd_sequence[idx]
                if str(hash(raw_other)) == tok:
                    lkp+=[raw_other]
            if tok not in set([m[0] for m in matches]):
                matches += [(tok, lkp)]
    # assert False, (processed, indices, matches)
    return processed, indices, matches

def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
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
        global_trv = data_cfg.get("global_trv", "")
        if global_trv:
            print(f"UserWarning global_trv parameter deprecated, use nothing instead.")
        trutrg = data_cfg.get("trutrg", "car") 
        canonization_mode = data_cfg.get("canonization_mode", "canonize")
        assert canonization_mode in ["canonize", "hash"], canonization_mode

        # TODO FIXME following is hardcoded; add to configs please
        pnctprepro = True
    else: 
        # the rest of the above variables are set at the end of load data for the non KB case
        pnctprepro = False

    # default joeyNMT behaviour for sentences

    tok_fun = list if level == "char" else (pkt_tokenize if pnctprepro else tokenize)

    src_field = data.Field(init_token=None,
                           eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN,
                           tokenize=tok_fun,
                           batch_first=True,
                           lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, 
                           eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN,
                           tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True,
                           lower=lowercase,
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
        # train_data has been loaded with normal extension (canonized files, e.g. train.carno)
        # dev/test_data will be loaded from non canonized files
        canon_trg = trg_lang # keep this for loss reporting (load dev/test data from here separately)
        trg_lang = trutrg

        train_kb_truvals = MonoDataset(
                                        path=train_path,
                                        ext=("."+kb_trv),
                                        field=("kbtrv", trv_field),
                                        filter_pred= lambda x: True
        )

        train_kb = TranslationDataset(
                                    path=train_path,
                                    exts=("." + kb_src, "." + kb_trg),
                                    fields=(("kbsrc", src_field), ("kbtrg", trg_field)),
                                    filter_pred= lambda x: True
        )
                                   
        with open(train_path+"."+kb_lkp, "r") as lkp:
            lookup = lkp.readlines()
        train_kb_lookup = [int(elem[:-1]) for elem in lookup if elem[:-1]]
        with open(train_path+"."+kb_len, "r") as lens:
            lengths = lens.readlines()
        train_kb_lengths = [int(elem[:-1]) for elem in lengths if elem[:-1]]
            
    # now that we have train data, build vocabulary from it. worry about dev and test data further below

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)

    # NOTE unused
    trg_vocab_file = data_cfg.get("trg_vocab", None)
    trg_kb_vocab_file = data_cfg.get("trg_kb_vocab", None)
    trg_vocab_file = trg_vocab_file if not trg_kb_vocab_file else trg_kb_vocab_file # prefer to use joint trg_kb_vocab_file if specified

    vocab_building_datasets = train_data if not kb_task else (train_data, train_kb)
    vocab_building_src_fields = "src" if not kb_task else ("src", "kbsrc")
    vocab_building_trg_fields = "trg" if not kb_task else ("trg", "kbtrg")

    src_vocab = build_vocab(fields=vocab_building_src_fields, min_freq=src_min_freq, max_size=src_max_size, dataset=vocab_building_datasets, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(fields=vocab_building_trg_fields, min_freq=trg_min_freq, max_size=trg_max_size, dataset=vocab_building_datasets, vocab_file=trg_vocab_file)


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

    if kb_task: #load dev kb and metadata; load canonized dev data for loss reporting

        dev_data_canon = TranslationDataset(path=dev_path,
                                        exts=("."+src_lang, "." + canon_trg),
                                        fields=(src_field, trg_field))

        dev_kb = TranslationDataset(path=dev_path,
                                exts=("." + kb_src, "." + kb_trg),
                                fields=(("kbsrc", src_field), ("kbtrg",trg_field)),
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

        test_data_canon = TranslationDataset(path=test_path,
                                        exts=("."+src_lang, "." + canon_trg),
                                        fields=(src_field, trg_field))
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
    src_field.vocab = src_vocab # also sets kb_src_field.vocab if theyre the same (variables point to same object)
    trg_field.vocab = trg_vocab

    if kb_task:
        # NOTE this vocab is hardcodedly built from the concatenation of train+dev+test trv files!
        # trv_path = train_path[:len(train_path)-train_path[::-1].find("/")]+global_trv
        # assert os.path.isfile(trv_path)

        trv_ext = "."+kb_trv

        trv_train_path = train_path+trv_ext
        trv_dev_path = dev_path+trv_ext
        trv_test_path = test_path+trv_ext

        assert os.path.isfile(trv_train_path)

        # try to make vocabulary exactly as large as needed

        # trv_vocab._from_file(trv_path)
        trv_vocab = deepcopy(trg_vocab)
        # FIXME only add this for source copying?
        trv_vocab._from_list(src_vocab.itos)

        trv_vocab._from_file(trv_train_path)
        trv_vocab._from_file(trv_dev_path)
        trv_vocab._from_file(trv_test_path)
        if canonization_mode == "canonize":
            # stanford data
            assert "schedule" in trv_vocab.itos
        # NOTE really important for model.postprocess: 
        # trv_vocab must begin with trg_vocab
        # to look up canonical tokens correctly
        assert trg_vocab.itos == trv_vocab.itos[:len(trg_vocab)]

        print(f"Added true value lines as tokens to trv_vocab of length={len(trv_vocab)}")
        trv_field.vocab = trv_vocab

    if kb_task:
        # make canonization function to create KB from source for batches without one 

        entities_path = "data/kvr/kvret_entities_altered.json" # TODO FIXME add to config
        entities = load_json(fp=entities_path)
        efficient_entities = preprocess_entity_dict(entities, lower=lowercase, tok_fun=tok_fun)


        if canonization_mode == "hash":
            # initialize with train knowledgebases
            hash_vocab = build_vocab(max_size=4096, dataset=train_kb, fields=vocab_building_trg_fields, 
                                    min_freq=1, 
                                    vocab_file=trv_train_path)
            hash_vocab._from_file(trv_train_path)
            hash_vocab._from_file(trv_dev_path)
            hash_vocab._from_file(trv_test_path)

            # assert False, hash_vocab.itos 
        
        # assert False, canonize_sequence(["your", "meeting", "in", "conference", "room", "100", "is", "with", "martha"], efficient_entities) # assert False, # NOTE
        # assert False, hash_canons(["Sure" , "the", "chinese", "good", "luck", "chinese", "food", "takeaway", "is", "on","the_good_luck_chinese_food_takeaway_address"], hash_vocab.itos) # assert False, # NOTE

        if canonization_mode == "canonize":
            class Canonizer:
                def __init__(self, copy_from_source: bool = False):
                    self.copy_from_source = bool(copy_from_source)
                def __call__(self, seq):
                    processed, indices, matches = canonize_sequence(seq, efficient_entities)
                    return processed, indices, matches
        elif canonization_mode == "hash":
            class Canonizer:
                def __init__(self, copy_from_source: bool = False):
                    self.copy_from_source = bool(copy_from_source)
                def __call__(self, seq):
                    processed, indices, matches = hash_canons(seq, hash_vocab.itos)
                    return processed, indices, matches
        else:
            raise ValueError(f"canonization mode {canonization_mode} not implemented")


    if not kb_task: #default values for normal pipeline
        train_kb, dev_kb, test_kb = None, None, None
        train_kb_lookup, dev_kb_lookup, test_kb_lookup = [],[],[]
        train_kb_lengths, dev_kb_lengths, test_kb_lengths = [],[],[]
        train_kb_truvals, dev_kb_truvals, test_kb_truvals = [],[],[]
        trv_vocab = None
        dev_data_canon, test_data_canon = [], []
        Canonizer = None

    
    # FIXME return dict here lol
    return train_data, dev_data, test_data,\
        src_vocab, trg_vocab,\
        train_kb, dev_kb, test_kb,\
        train_kb_lookup, dev_kb_lookup, test_kb_lookup,\
        train_kb_lengths, dev_kb_lengths, test_kb_lengths,\
        train_kb_truvals, dev_kb_truvals, test_kb_truvals,\
        trv_vocab, Canonizer, \
        dev_data_canon, test_data_canon


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
    # inherits from torch batch, not joey batch!
    def __init__(self, data=None, dataset=None, kb_data=None, kb_truval_data=None, device=None, canon_dataset=None):
        
        """
        Create a TorchBatchWithKB from a list of examples.
        Cant be put in .batch because of variable name clash
        between torchtext batch and joeynmt batch :(
        TODO FIXME document
        data is batch_with_kb object from below
        canon_dataset is optional dataset for having dual targets: "trg" and self.canon_field
        """

        if data is not None:
            assert hasattr(data, "kb") 
            self.batch_size = len(data)
            self.dataset = dataset
            self.kb_data = kb_data
            self.kb_truval_data = kb_truval_data

            self.canon_dataset = canon_dataset
            if self.canon_dataset is not None:
                assert canon_dataset.fields == canon_dataset.fields, (canon_dataset.fields, canon_dataset.fields)
                # set trgcanon field 
                self.canon_field = "trgcanon"

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
                    # assert name!="trg" or self.canon_dataset is None, (batch, field.vocab.arrays_to_sentences(self.trg[0].numpy().tolist())[0])
            
            for (name, field) in self.kb_truval_data.fields.items():
                if field is not None:
                    truvals = [getattr(x,name) for x in data.kbtrv]
                    preprocessed = field.preprocess(truvals)
                    processed = field.process(preprocessed, device=device)
                    setattr(self, name, processed)
                    if len(truvals) <= 5: 
                        print(f"matches for unk: \
                            {[(unk[0], field.vocab.stoi[unk[0]]) for i, unk in enumerate(truvals) if processed[i,1] == 0]}")

                        """
                            print(f"matches for unk: \
                                {[(unk[0], field.vocab.itos[lowest_med_match(unk[0], field.vocab.itos)[0]]) for i, unk in enumerate(truvals) if processed[i,1] == 0]}")
                        """

            for (name, field) in self.kb_data.fields.items():
                if field is not None:
                    # kb = [["@DUM"]] #dummy token (both for key and value)
                    kb = [getattr(x,name) for x in data.kb]
                    setattr(self, name, field.process(kb, device=device))
                else:
                    raise ValueError(field)
            if self.canon_dataset is not None:
                for (name, field) in self.canon_dataset.fields.items():
                    if name == "trg" and field is not None:
                        canontrg_batch = [getattr(x, name) for x in data.canontrg]
                        setattr(self, self.canon_field, field.process(canontrg_batch, device=device))
                        # assert False, (batch, field.vocab.arrays_to_sentences(self.trgcanon[0].numpy().tolist())[0])

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
        self.canontrg = [] # holds list with examples where example.trg is canonized

def dummy_KB_on_the_fly(trg_voc, kb_fields, kbtrv_fields):
    # called in data.batch_with_kb and again in helpers.store_attention_plots
    
    # try to get the raw token in the source that was replaced by '@event' as subject for the key
    subject = "dummySubj"
    relation = "dummyRel"
    value = "dummyVal"

    dummy_kb = [
        data.Example.fromlist(
        [[subject, PAD_TOKEN, relation], [relation]], #  hardcoded KB structure
        fields=list(kb_fields.items())
    )]
    dummy_kbtrv = [
        data.Example.fromlist(
        [[value]], 
        fields=list(kbtrv_fields.items())
    )] 

    return dummy_kb, dummy_kbtrv


def create_KB_on_the_fly(src_seq_str, trg_voc, kb_fields, kbtrv_fields, canonization_mode):
    # called in data.batch_with_kb and again in helpers.store_attention_plots

    src = src_seq_str
    relations, indices, matches = canonization_mode(src) # canonized source, try to make KB out of this 
    del matches # TODO use matches instead of this implementation below
    
    rels_vals = dict()
    prev_target = None

    for i, raw in enumerate(src):
        canon_idx = indices[i]
        relation = relations[canon_idx]
        if relation != raw:
            if relation not in rels_vals.keys():
                rels_vals[relation] = [raw]
            else: # entry already exists 
                if canon_idx == prev_target and rels_vals[relation][-1] != raw: # multi word expression, e.g. 'the' '11th' => '@date'
                    rels_vals[relation] += [raw]
                else: # multiple occurrences of e.g. @date  how to handle this?? XXX
                    pass
        prev_target = canon_idx
    
    #  dangerous heuristic requiring internal knowledge of everything in the universe

    EVENT = "@event"

    # try to get the raw token in the source that was replaced by '@event' as subject for the key
    subject = rels_vals.get(EVENT, [])
    if not subject:
        # if list is empty (no subject found in source), 
        # set dummy subject
        subject = ["dummySubj"]
    assert type(subject) == list, (type(subject), subject)

    on_the_fly_kb = [
        data.Example.fromlist(
            #  this reads [relation] using the first (source) field
            # FIXME TODO what does signature of data.Example.fromlist actually look like
        [subject + [PAD_TOKEN, relation[1:]], [relation]], # remove @ from @relation
        fields=list(kb_fields.items())
    ) for relation, _ in rels_vals.items() if not trg_voc.is_unk(relation)] #  replace 'False' by this to get on the fly creation again

    # input({" ".join(v): lowest_med_match(" ".join(v), kbtrv_fields["kbtrv"].vocab.itos, return_idx=False, topk=5) for r, v in rels_vals.items()})

    # FIXME some values like 'dinner' for some reason not pprcd even though theyre in the vocab.itos ???

    # assert set([" ".join(val) in kbtrv_fields["kbtrv"].vocab.itos for val in rels_vals.values()]) == {True},\
    #     []

    on_the_fly_kbtrv = [
        data.Example.fromlist(
        [[" ".join(val)]], # FIXME hardcoded KB structure
        fields=list(kbtrv_fields.items())
    ) for rel, val in rels_vals.items() if not trg_voc.is_unk(rel)] # FIXME replace 'False' by this to get it on the fly creation again

    """
    try:
        input(f"in on the fly creation: {[ex.kbtrv for ex in on_the_fly_kbtrv]}\n\tCancel (Ctrl+C) to see info\n\tEnter (Enter) to continue")
    except KeyboardInterrupt as e:
        input(f" rels_vals: {rels_vals}")
    """

    return on_the_fly_kb, on_the_fly_kbtrv


def batch_with_kb(data, kb_data, kb_lkp, kb_lens, kb_truvals, c=None, canon_data=None, max_chunk=64):
    # TODO document this hackiest of generators
    # minibatch.kb length, adds it to this attribute and yields
    # elements from data in chunks of conversations

    # max_chunk is maximum batch size if KB is the same for more than that many examples

    dstc2 = "r_phone" in kb_data.fields["kbsrc"].vocab.itos

    minibatch = KB_minibatch()
    current = 0
    last_corresponding_kb = 0 if not dstc2 else 1 # dstc2 has global KB as first KB so skip this
    kb_len = 0
    chunk = 0

    for i, ex in enumerate(data):

        try:
            corresponding_kb = kb_lkp[i]
        except:
            corresponding_kb = 0
            # assert False, kb_lkp # using different lkp file extension than expected?


        if corresponding_kb != last_corresponding_kb:

            print(f"minibatch #{i}; future ex.trg: {ex.trg}")
            yield minibatch
            minibatch = KB_minibatch()

            # sum over last kb and all inbetween that and current one (excluding current)
            # sometimes a KB is skipped
            current = sum(kb_lens[:corresponding_kb])
            chunk = 0 # reset chunk batch size

        elif chunk >= max_chunk:
            yield minibatch
            minibatch = KB_minibatch()
            chunk = 0

        # print(ex.trg, kb_lens, corresponding_kb)
        kb_len = kb_lens[corresponding_kb]


        minibatch.kb = kb_data[current:current+kb_len]
        minibatch.kbtrv = kb_truvals[current:current+kb_len]

        if not minibatch.kb:
            assert kb_lens[corresponding_kb] == 0
            input((i, corresponding_kb, kb_len))
        else:
            input(([ex.kbsrc for ex in minibatch.kb], len(minibatch.kb)))

        if len(minibatch.kb) == 0:
            # this is a scheduling dialogue without KB
            # try to set minibatch.kb, minibatch.kbtrv in a hacky, heuristic way by copying from source FIXME TODO XXX
            kb_empty = True
            if c is not None and c.copy_from_source: 
                otf_kb, otf_kbtrv = create_KB_on_the_fly(ex.src, data.fields["trg"].vocab, kb_data.fields, kb_truvals.fields, c)
                if len(otf_kb) > 0:
                    kb_empty = False
                    minibatch.kb = otf_kb
                    minibatch.kbtrv = otf_kbtrv 
            if kb_empty == True: # kb still empty even after maybe trying to copy from source
                dummy_kb, dummy_kbtrv = dummy_KB_on_the_fly(data.fields["trg"].vocab, kb_data.fields, kb_truvals.fields)
                minibatch.kb = dummy_kb
                minibatch.kbtrv = dummy_kbtrv

        chunk += 1
                
        assert len(minibatch.kb) == len(minibatch.kbtrv), \
            ([x.kbsrc for x in minibatch.kb],[x.kbtrv for x in minibatch.kbtrv])

        minibatch.append(ex)
        if canon_data is not None:
            try:
                minibatch.canontrg.append(canon_data[i])
            except Exception as e:
                print([len(thing) for thing in (canon_data, data)])
                print([[ex.trg for ex in thing] for thing in (canon_data, data)])
                raise e

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
        print()
        print(f"example number: {i}")
        print()
        print(f"kb_lkp array neighborhood of 2 (center item is corresponding_kb=kb_lkp[{i}]): {[(ex,kb) for ex,kb in enumerate(kb_lkp)][i-2:i+3]}")
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
        # there's no more data examples to append to this last minibatch, yield it
        yield minibatch

class KB_Iterator(Iterator):
    """
    Iterates over a kb in parallel to the dataset
    individual batches should have an attribute .kb
    which is equal to the corresponding kb Tensor (m x 3)

    for the moment,
    sorting and shuffling is ignored and not done
    """

    def __init__(self, dataset, kb_dataset, kb_lkp, kb_lens, kb_truvals,\
        sort_key=None, device=None, batch_size_fn=None, train=True,\
        repeat=False, shuffle=None, sort=None, sort_within_batch=None,\
        c_fn=None, canon_data=None):
        """
        takes additional args:
        :param kb_data: monodataset created in load_data from kb 
        :param kb_lkp: list of association lookups between dataset examples and \
            kbs in kb_data
        :param kb_len: list of kb lengths within kb_data
        FIXME document

        
        :return data_iter: data iterator object where each batch
        has an associated knowledgebase with it 
        """
        batch_size=1 #TODO remove batch_size specification /look at Iterator implementation of init to see if this does anything
        
        super(KB_Iterator, self).__init__(  
                                            dataset, batch_size, sort_key=None, device=None,\
                                            batch_size_fn=None, train=True, repeat=False, 
                                            shuffle=None, sort=None, sort_within_batch=None
                                         )
        self.kb_data = kb_dataset
        self.kb_lkp = kb_lkp
        self.kb_lens= kb_lens
        self.kb_truvals = kb_truvals
        # input(f"trv_vocab: {kb_truvals.fields['kbtrv'].vocab.itos}")
        self.c = c_fn # canonization function to canonize source incase KB empty
        # FIXME find out how not to assign this TODO FIXME XXX this is big on optimization
        self.canon_dataset = canon_data # used as secondary canonized dataset in valid/test for loss calculation 

    def data(self):
        return self.dataset

    def canon_data(self):
        return self.canon_dataset
        
    def create_batches(self):
        self.batches = batch_with_kb(self.data(), self.kb_data, self.kb_lkp, self.kb_lens, self.kb_truvals, c=self.c, canon_data=self.canon_data())
    
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
                batch = TorchBatchWithKB(minibatch, self.dataset, self.kb_data, self.kb_truvals, self.device, self.canon_data())
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
        fields = [('src', field)] if not type(field) == type(())\
             else [field] # field is already a tuple (name, attr)

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
                    shuffle: bool = False,
                    canonize = None,
                    canon_data: TranslationDataset = None) -> Iterator:
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
    :param c_fn: canonization function to create KB if KB empty
    :return: torchtext iterator
    """
    
    if train: # TODO remove this option entirely?
        # optionally shuffle and sort during training
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens, kb_truvals,\
            repeat=False, sort=False, 
            train=True, shuffle=shuffle, c_fn=canonize, canon_data=canon_data)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = KB_Iterator(dataset, kb_data, kb_lkp, kb_lens, kb_truvals,\
            repeat=False, train=False, sort=False, sort_within_batch=True,c_fn=canonize,canon_data=canon_data)

    return data_iter

def med(a, b):
    # calc minimum edit distance between strings a, b
    m, n = len(a), len(b)
    d = np.zeros((m,n))
    
    for i in range(m):
        d[i,0] = i
    for j in range(n):
        d[0,j] = j
    for j in range(n):
        for i in range(m):
            if a[i] == b[j]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i,j] = min(d[i-1,j]+1, # del
                        d[i,j-1]+1, # ins
                        d[i-1,j-1]+substitutionCost # sub
            )
    return d[m-1,n-1]


def lowest_med_match(query, keys, return_idx=True, topk=1, short_penalty=False):
    query = query.lower()
    keys = [key.lower() for key in keys]
    scores = [med(query,key) for key in keys]

    if short_penalty:
        scores = [score+len(query)-len(key) for score, key in zip(scores, keys) if len(query)-len(key) > 0 ]

    # best is argmin of scores

    # sorts ascending
    sort_by_scores = sorted(list(zip(scores,range(len(keys)))), key=lambda score_and_key: score_and_key[0])

    topk -= 1 #start counting at 0

    topk_scores_keys = [(score, key) for score, key in sort_by_scores[:topk]]

    if return_idx:
        return [top[1] for top in topk_scores_keys]
    else:
        return [keys[top[1]] for top in topk_scores_keys]

