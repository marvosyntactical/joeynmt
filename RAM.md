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

4. load kb during testing
  * like how its done in validate


---

# Active Workspace

### Technical TODO:

This is a general list of minor technical TODOs that can be done without thinking. Thoughts are nevertheless jotted down unorderedly:

* figure out how to reconstruct tokens for debugging within training/model/decoder
  * something like: *self.trg\_embed*.lut[array] 
  * also look at *array\_ to\_ sentence*
* look at torchtext.dataset.sort\_key within load\_data: are my batch attributes shuffled during trai/val/test???

## _```Current issue```_:
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
-> use mono dataset tensor for true kb values and add kb\_truval attrib to batch
-> batch now has attributes: *src*, *trg* and _kb-src_, _kb-trg_ and _kb-truval_




























---

# Issues Archive

## _Old Issue_:
### 21.02. Brainy Juice:

* during decoder.forward, kb with triple items has to be attended to => several options:
	1. batch has multiple scenarios, maybe even standard batch size => use lookup table to pass needed scenario KBs to decoder attention
	2. batch has one scenario, variable batch size (dialogue length) => iterate over scenarios in parallel to train\_iter and pass one KB per batch/forward pass to attention ```CURRENT SOLUTION```
	3. batch\_size = 1 => use lkp table to take relevant scenario 
In the current solution, batch multiplier could be increased; on average the batch size is currently ~3; due to how small the dataset is batch size should have an impact but not really matter in training time





