* skim through relevant outputs by filtering for "proc\_batch"
* outputs are printed for last element of every batch (because for the last element of the batch, the source contains all the info)
* in 2, without adding v to outputs, you can see hypotheses still contain lots of kb\_values randomly suggested

-> need to find alleviation for sparsity?

Latest TODO: two options lead to two different problems:
in load\_data:
fields=vocab\_building\_src\_fields <- should be correct
leads to completely jumbled up kb\_keys entries: wrong token lookup itos
fields=vocab\_building\_trg\_fields
leads to lots of unk in kb\_keys
Intended: fields=vocab\_building\_src\_fields(==("src", "kbsrc"))
should make kb\_keys tensor filled with correct !=unk tokens

length of src and trg vocab (2717, 2695) suggests that vocab doesnt read in the kb vocab... however ../runs/Vocab.txt clearly prints out that they are added!
