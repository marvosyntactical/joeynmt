_Work rhythm_:

  0. Use **tmux** with dedicated windows (top right ram, lower right repo navi; left training); code in **VS**
  1. Use daily RAM file (copy this from last day, leave top the same, change only current issue (pick one from the TODO list); jot down ideas for open issues below)
  2. Only commit if training works. **REALLY TRY** to end the day with a commit/finished thought regarding the current issue. Think about what to tackle next. Spend too much resources on a problem? Ask Artem.
  3. Start the day by **REALLY** thinking about how to solve the current issue: planning ahead seems to save a lot of trouble later :^)
  4. Adopt consistent markdown style:*bold joeynmt/torch/existing variables and words* and _cursive  eric et al variables and words added to the system during my implementation_, **BOLD CURSIVE CAPS FOR REALLY IMPORTANT STUFF**

# Open Questions and Ideas for the Future:

* Is the entire model trained on just these datasets? e.g. kvr has 2.4k pairs 
* Very heterogenous dataset: Even within domain, e.g. 'scheduling', different task types, e.g. time request (like weather, info retrieval) but also making an appointment => filter out everything that is not KB retrieval?; other than that, the info retrieval is essentially the same for all 3 domains => good!
* Trg vocab goes from 40k to 112k by adding knowledgebase..unusual but should be doable with how small the dataset is
* how does kv attention actually work for normalized (-> triples) entries? how does attention understand on the request "Wheres the nearest cafe?" to look up the poi\_type value for starbucks among others, see cafè; and then learn to look up the distance and address for the same subject? the keys the attention sees are NOT THE SAME, they conflate both subj and relation?!? does the magic lie in the successive kb attention queries from one decoder unrol step to successive ones? is the key rep expected by the rnn cell to be incorporated into the cell state? doesnt this mean we need to track conversation long (and not just seq2seq query - response isolated examples) history either by concatenating all previous utterances or by somehow using the last previous hidden states?
* understand: where does loss come from? => dont we need to call our loss function on the postprocessed (de-canonicalized) generated trg sequence to learn? or is this not necessary? would be interesting to bleu, but does it even matter there? does it matter to Xent?
* does it make sense to use entity F1 as supplementary validation metric?
* meaning and impact of LSTM forget gate bias (=> Pham et al.)



# Issues ```TODO```:

This table is for the project wide software engineering / research considerations.
These will have to get resolved someday. Unordered thoughts also jotted down:

1. filter out unvalued kb entries (wait for Artem's response)


2. NER/linking
  * linking is different from _kb-canonization_ and later lookup of kb values!
  * it seems like ALL (not just KB) named entities are linked
  * pick NER (StanfordNER prob used by authors -> finally get this to work) vs (SOTA NER) (Ask Artem: which is higher: reproducing results or achieving good ones)
  * at what point to we link to _ner-canonical-name_? probably before producing vocab file; also save occurence probabilities in train data (only utterances?also kb?) for later backgeneration
  * so during preprocessing, before generating any of the files used by *load_data*, let NER run over the utterances (and kb? here, referring expressions have no context. this impacts choice of NER system) and cluster, saving clusters and occurrence rates. Clusters are then named, this _ner-canonical-name_ is substituted for all referring expressions. Only now, *src\_* and *trg\_vocab* are generated.


3. merge updated joeynmt back
  * rather sooner than later....


---

# Active Workspace

### Technical TODO:

This is a general list of minor technical TODOs that can be done without thinking. 

* fix kbtrv:
* fix build\_vocab issues: train kb contains a lot of unk tokens (esp relations are unk)
* understand decoder unroll: why always the same unroll steps (trg\_input)... unroll steps is exactly the number of to be generated tokens, or is it an upper bound? does the decoder stop on eos? 
  * (decoder does stop on eos in latest joeynmt commit)
* find out what shuffle in training data means: intra or supra batch?
  * -> intra batch of course
* import and use tensorboard writer again
* batch convos with same kb together; ! filter unvalued entries !
* traffic info: default category in this domain: poi\_type, not poi!

####### Optimization
* rewrite model.run\_batch post processing to use numpy funs instead of double for
* figure out how to make joeynmt.vocabulary.Vocabulary object serializable for optional saving in joeynmt.data.load\_data 
  * -> (not important since vocab is small now)


## _```Current Issues```_:

### 15.05.20 implement kb for transformer

steps:

1. merge generator branch back
2. test backwards compatibility for rnn without kb task
3. test backwards compatibility for transformer without kb task

4. implement kb for transformer:
* interface wise just need to generate kb\_probs somewhere within transformerdecoder
-> can kb\_probs calculation actually be moved to generator?
-> RNN: can kb\_probs be calculated in forward after forward\_step() unroll loop?

### 08.05.20 debug kb batch matchup

batches have random kbs??
look in:
* data.batch\_with\_kb
* preproc: data/scripts\_kvr/ parse\_kvr\_json.py and normalize\_scenarios.py


### 19.04.20 canonize target to same resolution as kb values 

* for kvr\_attention to learn, its output like meeting\_time must be contained in the same form in target sequences; in the training data they occur as e.g. "4", "pm" though!
* step 1 (mostly done, TODO debug canonize for some quirks (look at dev/train/test.carnon)): use kvret\_entities.json to canonize target (very low granularity, e.g. "4", "pm" -> "@time" (instead of "meeting\_time")
  * -> TODO TODO TODO FIXME FIXME FIXME
* step 2 : map knowledgebase values (medium granularity (e.g. "meeting\_time") to low granularity ("meeting\_time" -> "@time")
  * -> Done
* step 3 : in step 2, keep info about replacement somewhere and do an inverse lookup later (recover "meeting\_time" from "@time" and from thereon "4" "pm" (kbtrv))
  * in search.py, need to recover correct token if next\_word is a knowledgebase token / canonized
  * => rewrite vocabulary.Vocabulary to check for @ at beginning of words and put them to the end and have an attribute Vocabulary.canon\_start\_idx
  * => in search.py, after decoding for one step, do this check: ```if next\_word >= trg_vocab.canon\_start\_idx:\\ next_word = highest attended value in knowledgebase out of matching canon values```
  * -> All Done

This is not what the authors did and it is questionable whether this attention based recovery is sufficient to learn.

* in kbv, (files with new\_york\_wednesday formatting, prepend @ for vocabulary check)

-> it still seems like some knowledgebases are matched with the wrong batch....

### 07.04.20 training on GPU

* 63 epochs after 520 minutes => 8.2 minutes per epoch
* validate every 100 examples
* TOK/SEC increases per epoch from 200 to 6600
* testing should work (beam search), test again with saved checkpt

---

# Issues Archive

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





