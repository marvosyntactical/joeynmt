# Ideas
After reimplementing Eric et al (2017) and transferring the central idea to transformers (Vaswani et al. 2017) it's sensible to start thinking about possible contributions in this work. Those could be one or several of:


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


