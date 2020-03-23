This is a fork of https://github.com/joeynmt/joeynmt/

# Reimplementation of [Key-Value Retrieval Networks for Task-Oriented Dialogue](https://arxiv.org/pdf/1705.05414.pdf)
This is where one day, my bachelor's thesis will hopefully be completed.

Check *RAM.md* for current problems, ideas and questions

---

#### Talks with Artem

##### Talk on Mon 09.03.:

Issue: Some knowledgebases are empty
Solution: Add default dummy entry to all knowledgebases.
Would a sensible alternative be to just not forward the kvr_attention for these batches?

##### Talk on Fri Feb 21:

Goals were set:

### Primary task
 Reimplement the kvr model, use same dataset, achieve/compare/criticize results
### Secondary task
 Adapt the model for transformer
### Tertiary/Additional
 Depending on if necessary, enrich dataset, pretrain model OOD etc; possibly look at chatbots (probably not) 


---

#### Related resources:

Other datasets:

* https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
* https://research.fb.com/downloads/babi/ 


* Take a look at [JoeyNMT nonworking implementation](https://gitlab.cl.uni-heidelberg.de/zoll/swp-joeynmt/)
* Look at [Keras Implementation](https://github.com/sunnysai12345/KVMemnn)
