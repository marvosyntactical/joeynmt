## Small Technical _TODOs_ & questions

##### Optimization
(later)

# _OPEN ISSUES_


### 28.09.30 current model issues:

* unk from copying *DONE*
* bad performance *TEST*
* try: feed utilities; actually use feeding layer, aggregate vector in kth hop?
* try: feed all utilities
* compare with default version


### 26.09. idea for multihop 

* layerwise weight tying as in weston et al: next module is same module 
* pass hidden state to next module instead of utilities:
* batch x kb\_curr\_dim x hidden instead of batch x 1 x kb
* *TODO* test: can I still mask between hops???? should annihilate everything?
* *TODO* test: is the aggregate kb hidden tensor better outside or inside the loop?


### 11.09. canonization levels

* 5 pm 
* meeting\_time
=> do linking on canonized source and target *TODO*
* time

### 10.09. ent f1 calc

*TODO* fixme

### 21.09. questions for eric et al

* did you do autoregressive training?
* canon level

### 21.09. grid search preparations

*TODO*
Code:

* correct entity F1 reporting
* implement multiheaded KVR attention for transformer
* what happens with k\_hops == 2 and input\_feeding == False? dont I need previous utilities?
* fix transformer encoder???

Test runs:
* 2D KB plotting 

### 09.09. grid search hyperparams

These hyperparams are all orthogonal:

* eric et al replication: RNN, 1 hop, no kb input feeding, autoregressive, @meeting\_time level, kb dim 1  
* metric reporting: raw, canonized

====== GRID SEARCH OVER THESE ======
* 3 multihops: 1 , 2 , 3
* 3 kb input feeding: False, ff, rnn
* 2 teacher\_force: no, yes
* 2 architecture: RNN , rnnTF
====== END GRID SEARCH OVER THESE ======

3 x 3 x 2 x 2 x 12 x 1/6  = 72 stunden = 3 tage 

* 3 training data level: @time, @meeting\_time *TODO*, 3 pm 
* 2 scheduled sampling: invsigmoid, linear
* 2 tftf, rnntf

* 3 kb encoding: 1D, 2D, positional
* 2 kb embedding: source,  separate
* 2 copy\_from\_source: True, False
* 2 kb key rep: with values, without values


### Empty scheduling KBs:
* no knowledgebase in half of scheduling dialogues => nothing to replace canonicals with
* => in data.py, for minibatches with empty kb, just canonize source and add that as knowledgebase?
* (e.g. "Make an entry for dinner on the 6th at 7 pm with my sister."
* =>    "Make an entry for @event on @date at @time with my @party ."
* =>    (dinner event dinner), (dinner date the 6th), (dinner time 7pm), (dinner party sister)
* make these the knowledgebase, so they can be attended over
✔️
*TODO* TEST this

### 31.08.20 stuff to add to config

* copy from source? bool✔️
* knowledgebase encoding: "2d"✔️, "positional" ✔️
* separate KB embedding table ✔️

### 29.08.20 plotting issues

* made rShort, tShort commands for running short valid locally (r=Rnn, t=transf)
*TODO* update .blablaSHORT data using NODEFAULT data
* look at first kb there
* misalign: on the fly empty kb in first example of dev set has
 - attention matrix: 128 x 5 # find out where this happens (put asserts in model run batch)
 - remade on the fly kb: 128 x 2
 - on the fly kb: 128 x 2
*TODO*


### 26.08.20 start writing

Outline:

Introduction
Related Work
Background
Architecture
Dataset
Experiments
Conclusion

---

### 10.09.20 DUMMY entries vs deciding to copy from source

* add copy from source option ✔️
* if KB empty, dont use attention module✔️

### 08.09.20 Transformer

* find a working vanilla implementation; Options:
1. use RNN version of kvr attention *BRANCH TODO NOW*
2. put kb-trg attention's output thru some sort of feedforward layer
3. bruh
* add other stuff:
1.  multiple hops
2.  multiple dimensions

---

### 28.08.20 cheat version (bleu on canonized)

* Canonization level: can meeting\_time level be achieved? need linked target data: how to map any 5 pm to meeting\_time or 20 Main Street to Pizza\_My\_Heart\_Address
*TODO* link to pizza hut address level !!!!!

---

### 26.08.20 scalability *TODO*

* calculate/profile attention runtime
=> linear in hops
=> linear in dims
=> same as bahdanau
* mix in-domain KBs together (refactor preproc scripts first...)

---

### 26.08.20 enrich KB entry encoding *TODO*

Artem encoding idea:
Instead of choosing one KB attribute per domain as
KB entry key, e.g. "event" for scheduling;
just jam all info into the key representation, and determine what 
is subject and what is relation via 
PositionalEncoding (make keys fixed size and add some exp+trig trickery)

* in preproc, create copies of kb with all infos in the keys (in same order everytime)
* if config["kbattencoding"] == "positional", pipe kb keys thru transformer\_layers.positional\_encoding 

Another encoding idea:
Make attention 2-headed, one head for subject (sum of all attribute embeddings);
a second head for relation 
(=> only 5 attributes for scheduling/traffic, ~10 attributes for weather)
Select key as combination of highest attended

* in preproc, create copies of kb with all attributes in the keys 
* if config["kbattencoding"] == "twoheaded": 
 - in decoder init, give twoheaded=True as arg and init attention module with 2 heads
 - transform kb key tensor in model.preprocesskb, adding extra dim for last elems and striding along this dim until they repeat themselves

*TODO* look at results


---

### 26.08.20 artem questions *TODO*

Architecture/implementation:
* name/concept/citation for the idea of training on simplified data, then postprocessing on val/test => label shift

Dataset stuff to report/issues:
* metric to show train/val/test dataset overlap too big?
* dataset simplicity (is there a simple measure for this?)
* in dataset section, discuss misleading design decisions:
KBs unambiguous as designed; not like real world => experiment

Metrics stuff :
* include phi coefficient with TN in metrics / validation 
* ablation study with bleu without "Youre welcome", "Have a nice day", "How can I help you today?" etc etc
* beam search should work now ✔️ 



### 31.08.20 recurrent multihop issues
=> seems to work okay, but kb attentions dont get plotted anymore
*TODO*
=> is attention state actually saved from step to step as done by jason weston et al?

---

# Issues Archive

## Old Issue

## Old Issue

## Old Issue

## Old Issue

### 09.09. different embedding table for kb keys
✔️



## Old Issue
### 08.09.20 how does attention mechanism learn to distinguish gas station from restaurant?

* the model doesnt get a loss signal because of the canonization: @poi vs @poi
* probably through a combination of embedding similarity and sentence structure
* try different embedding table (than src vocab) for kb keys✔️
* if it still works, embedding similarity is not necessary



## Old Issue
### 28.08.20 wikiBIO dataset


#### Problems:

##### 0: What is input to generate first sentence?
Just title + KB ?

##### A: Knowledgebases (infoboxes) dont contain all entities used in bios 

###### Approach 1:
* use NER system to link/label entities in bios without looking at infobox
* create huge json file with lookups like @birthday: [jan 1st 1900, jan 2nd 1900,...]
* only be able to replace generated labels that happen to appear in KB (get fd)

This is more like a real world task
Model will be incentivized to generate labels that appear in the KB because of training signal
But: Because all of the training data was labelled, the only way to be creative is to generate labels that dont appear in the infobox still


###### Approach 2:
* take values from box and link these to entities in bios
* replace these entities with infobox categories as labels

In this case the model doesnt actually make any choice in the knowledgebase.
Even if Problem B is adressed, the model only needs to choose an entry (trivial) and selecting an attribute should be unambiguous again

This might be worth it, if we care about showing the KVR Attention can select which entry to attend to

##### Problem B: Knowledgebases not ambiguous enough (want more diversity)

Collect many infoboxes entries in large knowledgebases

Problem: Heterogenous data structure across infoboxes 
result: lots of unassigned entries and loose ends


## Old Issue
### 28.08.20 scheduled sampling

implement scheduled sampling:

yaml:
add hyperparams:
scheduled\_sampling\_type: linear, exponential, invsigmoid (default) 
scheduled\_sampling\_k: value of k (float) # allowed range depends on sampl type
scheduled\_sampling\_c: value of c (slope of linear schedule)
✔️

#### TrainManager:
* initialize with 
self.scheduled\_sampling(type,k,c) => function that only depends on i
* keep track of num of minibatches i across epochs
* for each batch: calculate e\_i
* call glfb with e\_i
✔️
#### get\_loss\_for\_batch:
args:
e\_i: float = 0.
pass this to greedy search if > 0.
✔️
#### greedy\_search:
args: 
batch.trg: Tensor = None, e\_i: float = 0.
if received arg batch.trg, then select previous y with probability e\_i
from batch.trg or model prediction
✔️




## Old Issue
### 15.08.20 reccurent multihop

implemented in multihop branch in attention.py and merged back into master ✔️

implemented as "k-hop": 

```
utilities\_k = empty tensor
do k times:
   input feed (concatenate) utilities\_k with keys
   calculate utilities\_k (energies) as normal
```

special case: k=1: 1hop is equiv to default (same results) ✔️

=> put khop into config ✔️
=> do 2, 3 hops ✔️

* do a while loop and hop as many times as needed until suffctly confident?




## Old Issue
### 17.08.20 preprocessing 

Pipeline:
* remove double "home" in traffic KBs ✔️
* refactor data/scripts\_kvr/ into one script; go.sh ✔️
* (batch convos with same kb together) 



## Old Issue
### 15.05.20 implement kb for transformer *WIP*

1. merge generator branch back
-> Done
2. test backwards compatibility for rnn without kb task
-> Done, Check
3. test backwards compatibility for transformer without kb task
-> Done, Check

4. implement kb for transformer:

multi head kb attentions now get calculated in new MultiHeadedKbAttention class:
* works like MultiHeadedAttention except no matmul with values
* instead returns kb\_probs tensor of shape B x M x KB which is passed along until the generator, where its applied to outputs using kb\_values indices just like the recurrent case
* => looks like generator successfully works for rnn and transf case

Bad Results => Something's wrong:
* Hyperparams/Config?
* Implementation?
* learns slowly/poorly investigate hyperparam correctness such as label smoothing
✔️
Alternative Version:
* Single attention pass with query=h2\_norm of last layer works better
✔️
MultiHeadedKBAttention:
* figure out if possible to make this more like vanilla transformer (nativity)
* what to do with heads? atm: sum (artem: information loss!)

*TODO* Scheduler should be Noam



## Old Issue
### 07.04.20 training on GPU

* 63 epochs after 520 minutes => 8.2 minutes per epoch
* validate every 100 examples
* TOK/SEC increases per epoch from 200 to 6600
* testing should work (beam search), test again with saved checkpt

TODO for actual gpu training:
* import and use tensorboard writer again
*DONE*



## Old Issue
### 29.08.20 perplexity 

perplexity should be going down instead of up
it should be upper bounded by output vocab

=> this happens because valid loss is reported on uncanonized data (hard)

=> solution:
* canonize dev.car into dev.carno
* make a new TorchBatchWithKB attribute validtrg for the valid\_iter KB\_Iterator 
* use this in model.get\_loss\_for\_batch if torch no grad and its available

*SOLVED*



## Old Issue
### 10.08.20 jump back in: TODOS

1. whats not working in stock impl?

* choose best model based on bleu not ppl??
=> early stopping metric in config set to eval\_metric or anything

1a. find out whats not working with postprocessing: 

Plotting:
* valid\_kb.kbtrv ist LEER! (=> versuche plotting mit kbtrg zu reetablieren)
 TorchBatchWithKB.kbtrv ist schon immer leer!!!
* dev.lkp ist FALSCH (=> entspr. teil der preprocessing pipeline durchgehen)
* does kb attention actually sum to 1 ???
  (should not be the case because for KBs with only dummy token, that should have high probability and always be favored to be output)


## Old Issue
### 12.08.20 data formatting issues

* canonical target data has @poi\_type, @distance
* knowledgebases have NO @poi\_type (used as key), @poi\_distance 

*DONE*

KB considerations:

WEATHER:

* canonical trg weather data not perfectly formatted:
- some temperatures missing

* canonical trg weather data contains: @weekly\_time @date
=> remove @weekly\_time from from kvret\_entities\_altered.json *DONE*
=> add date to weather weekdays *DONE*

TRAFFIC:

* canonical trg data contains @poi\_type (a third (!) as often as poi\_name), add to kb as "stanford express care+hospital=hospital"!
*DONE*


Postprocessing:

* "clear" (trv\_vocab.stoi[-1]) used as replacement for unmatched canonicals
switched replacement array to kb\_trv, now actual replacements should be used

* inner pp for loop token if check triggers on tokens that arent even in the hypothesis???
(according to prints)
=> assertion added; this doesnt trigger; investigate in next training output
* does kb attention actually sum to 1 ???



## Old Issue
### 11.08.20 postprocessing: failure to match canonicals
Valid/Test Postprocessing:

* (1. select top k => 2. restrict to matching => 3. choose top 1)
* to
* (1. restrict to matching => 2. select top 1)

*DONE*



## Old Issue
### 08.05.20 debug kb batch matchup

batches have random kbs??
look in:
* data.batch\_with\_kb
* preproc: data/scripts\_kvr/ parse\_kvr\_json.py and normalize\_scenarios.py

-> for debugging, create bash alias for config with no\_traffic/best.ckpt as load\_model ... then check hypotheses postprocessing quick

=> Done, dev\_kb\_len was wrongly assigned in training.py startup ..

## Old Issue
### 02.06.20 implement generator in preparation for transformer

-> mostly done, works for recurrent decoder

=> Done


## Old Issue
### 19.04.20 canonize target to same resolution as kb values 

* for kvr\_attention to learn, its output like e.g. "@meeting\_time" must be contained in the same form in target sequences; in the training data they occur as e.g. "4", "pm" though!

Steps:

1. use kvret\_entities.json to canonize target (very low granularity, e.g. "4", "pm" -> "@time" (instead of "@meeting\_time")
  * -> mostly done, TODO debug data/scrits/canonize.py for some quirks (look at dev/train/test.carnon) FIXME big time TODO 

2. map knowledgebase values (medium granularity (e.g. "meeting\_time") to low granularity ("meeting\_time" -> "@time")
  * -> Done

3. in step 2, keep info about replacement somewhere and do an inverse lookup later (recover "meeting\_time" from "@time" and from thereon "4" "pm" (kbtrv))

  * in search.py, need to recover correct token if next\_word is a knowledgebase token / canonized
  * => rewrite vocabulary.Vocabulary to check for @ at beginning of words and put them to the end and have an attribute Vocabulary.canon\_start\_idx
  * => in search.py, after decoding for one step, do this check: ```if next\_word >= trg_vocab.canon\_start\_idx:\\ next_word = highest attended value in knowledgebase out of matching canon values```
  * -> All Done

This is not what the authors did and it is questionable whether this attention based recovery is sufficient to learn.



## Old Issue
### 10.06.20 improve canonization 

1. optimize canonize.py
2. weather:
* split into temperature and weather type (determine how we can meaningfully split by looking at train.car lines: what do people choose to say?)
* give location info
* give weekday info (especially on today)
* meaningful weather split is probably: 
* (@temperature, @precipitation, @day\_of\_week, @weather\_location)

=> Create new *branch* before the dangerous following stuff:
* update knowledgebase files: split weather kbs as above
* while at it, also remove empty ("-") scheduling entries
* while at it, also add poi\_name : poi\_name to traffic entries 
* go through canonization pipeline to get proper new *.len* files

=> Done


## Old Issue
### 14.05.20 switch to generator

in preparation if implementing the transformer version, adapt generator style of decoder:

at the end of decoder forward pass, a generator class functions as the output layer


## Old Issue
### 10.04.20 fix kb trv tensor/vocab

fixed batch size being 11 (9+1+1) (probably data.batch\_with\_kb
fixed kbtrv vocab not loading (everything UNK)

## Old Issue
### 14.04.20 plot kb attentions

added plotting of kb attention

## Old Issue
### 06.04.20 implement v

Implementing V instead of V\_t

-> done, for documentation look at kvrretdecoder.forward()


### 02.04.20 Dialogue history to source:

add the entire dialogue history to source
problems:
* makes batches unwieldy one batch is one convo, with the batch size being equal to the last utterance (with src==entire history) and examples mostly pad

-> added dialogue history to src; 
-> utterances separated by \<dot\> token

Finished on 02.04.


## _Old Issue_:
### 23.03.20 Refactor KB:

Dimension, Dataset (from 1 to 2), vocab file, kb preproc

Update kb tensor to use multiple word embeddings instead of one per _subject_ and _relation_.

* that means list of tokens with padding
* find out dimensions
* add canon val to tensor (in kb preproc) (entries go from triple to quadruple)
  * or make the triple (_subj_, _rel_, _kb-canon-val_) and also pass list or dict of actual _values_
* change vocab used by _kb_ (currently: _trg\_vocab_)
  * two different vocabs!! *src* for subj_ and _rel_; *trg* for _val_ and _canonval_. _data.torchbatchwithkb_ can use *TranslationDataset* instead of MonoDataset!

-> use translation dataset tensor with trg equal to just canonical name; pass list of actual values through repo
-> build vocabulary for e.g. source from both train\_data.src AND train\_kb.kbsrc
-> use mono dataset tensor for true kb values and add kb\_truval attrib to batch
-> batch now has attributes: *src*, *trg* and _kb-src_, _kb-trg_ and _kb-truval_

Finished on April 2


## _Old Issue_:
### 21.02. Brainy Juice:

* during decoder.forward, kb with triple items has to be attended to => several options:
	1. batch has multiple scenarios, maybe even standard batch size => use lookup table to pass needed scenario KBs to decoder attention
	2. batch has one scenario, variable batch size (dialogue length) => iterate over scenarios in parallel to train\_iter and pass one KB per batch/forward pass to attention ```CURRENT SOLUTION```
	3. batch\_size = 1 => use lkp table to take relevant scenario 
In the current solution, batch multiplier could be increased; on average the batch size is currently ~3; due to how small the dataset is batch size should have an impact but not really matter in training time

Finished early March (when?)





