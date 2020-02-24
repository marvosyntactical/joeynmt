This is a fork of https://github.com/joeynmt/joeynmt/

# Reimplementation of [Key-Value Retrieval Networks for Task-Oriented Dialogue](https://arxiv.org/pdf/1705.05414.pdf)

#### Dataset questions / Implementation options

* Is the entire model trained on just these datasets? e.g. kvr has 2.4k pairs 
* Very heterogenous dataset: Even within domain, e.g. 'scheduling', different task types, e.g. time request (like weather, info retrieval) but also making an appointment => filter out everything that is not KB retrieval?; other than that, the info retrieval is essentially the same for all 3 domains => good!
* during decoder.forward, kb with triple items has to be attended to => several options:
	1. batch has multiple scenarios, maybe even standard batch size => use lookup table to pass needed scenario KBs to decoder attention
	2. batch has one scenario, variable batch size (dialogue length) => iterate over scenarios in parallel to train\_iter and pass one KB per batch/forward pass to attention
	3. batch\_size = 1 => use lkp table to take relevant scenario 
* In latter two approaches, batch multiplier can be increased; on average batch size in approach 2 would be ~3; due to how small the dataset is batch size should have an impact but not really matter in training time

##### Talk on Fri Feb 21:

* Primary goal: Reimplement the kvr model, use same dataset, achieve/compare/criticize results
* Secondary: Adapt the model for transformer
* Additional: Depending on if necessary, enrich dataset, pretrain model OOD etc; possibly look at chatbot 

### ```TODO```

- entity linking: at what point do we map entities? only knowledgebase entities? what entity linking system can be used?
- take a look at follow up papers:
	1. mem2seq:  todo



* https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
* https://research.fb.com/downloads/babi/ 


2. Look at [JoeyNMT nonworking implementation](https://gitlab.cl.uni-heidelberg.de/zoll/swp-joeynmt/)
3. Look at [Keras Implementation](https://github.com/sunnysai12345/KVMemnn)
