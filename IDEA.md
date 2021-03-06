# Ideas
After reimplementing Eric et al (2017) and transferring the central idea to transformers (Vaswani et al. 2017) it's time to start thinking about possible contributions in this work. Those could be one or several of:


1. Dialogue joint knowledge task
* as I say in inter\_fere section future work, have knowledgebase be a joint shared world knowledge for two networks to optimize together
* dataset 
  * MUST have: knowledgebase that is updated during convo (like "slots" in kvr), conversations that require use of a shared knowledgebase
  * SHOULD have: longer conversations, knowledgebase that warrants logical derivations (for logical inference), symmetric conversations (not usr, car (cant train on usr))

* if conversations are symmetric, have one network run over both sides of the convo 
* in current context and with current data, could for example also add a network TODO finish this sentence

-> probably undoable, but maybe one of the subtasks is sensible

-> probably requires data augmentation

2. API Calls (e.g. Logic Parser)
* could issue calls to tableau/other predicate logic parser
* in order for logic to be necessary to the system, the data would need to look different though
* slow, discrete and noisy training signal -> very bad+clumsy -> bad enough?

-> probably requires data augmentation



3. Data Augmentation
* before augmenting, actually use the entire data: make use of requested slots (e.g. instead of convo history as source, provide requested slots inside of the knowledgebase) 

Is there a way to enlargen the dataset for specific subtasks, e.g. to get more data for the calendar entry making task?


4. How scalable is the knowledgebase?
The main reason the authors probably kept the knowledgebase small was to make the task doable for turkers.
The task should still be doable (only take more time) if we inject knowledgebases with more (irrelevant?) examples (from the other knowledgebases in the dataset.) Explore how far we can go with this.


5. Other database operations via soft attention
The key value retrieval attention is an implementation of the database operation "search/retrieve". Could other database operations be implemented with soft attention?
These would be:

* Save: make new kb tensor entry: somehow find out which tokens in source should be summed to be the key, and which should be value. Since value is taken from source, target vocab must be subset of source vocab (best be the same for lookup table match...). Can attention learn which entries need to be made? The knowledgebase would need a training signal. Could e.g. make the model spit out newly saved key value pairs after EOS or some other special token; and add the correct key value pairs to the the target sequences 
* Change: three step process: find corresponding key and as above, figure out what the new value is and replace that value: 1.search 2. delete 3. save
* Delete: probably dont remove entry from kb tensor, instead replace with some sort of default token or add delete encoding to entry

In order for an attention to learn to correctly update the database with the above operations we need training data that gives info about updated data: 

* half of the "scheduling" domain in the KVR dataset uses the slots tracking! should be about 500 examples, in one third of which updates should actually happen, so ~150 examples

6. take a look once again at sequence to query transducers

* [WikiSQL Task/Dataset](https://github.com/lukovnikov/WikiSQL)
* [SQL-Net (2017)](https://arxiv.org/abs/1711.04436)
* [Seq2SQL (2017)](https://arxiv.org/abs/1709.00103)
* [TypeSQL (outperforms SQLNet)(2018)](https://arxiv.org/abs/1804.09769)


and more tasks (read these especially for presentation prep for 18.06.2020):

* [bAbI toy task set (2015)](https://arxiv.org/abs/1502.05698)
* [Memory Networks (2014)] (https://arxiv.org/abs/1410.3916)


