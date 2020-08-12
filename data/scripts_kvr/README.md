execution order:
with sys.argv[1] each of 'train', 'dev', 'test', i.e. e.g.:

$ python3 scriptname.py train
$ python3 scriptname.py dev 
$ python3 scriptname.py test 

1. execute the following files in the following order:

parse\_kvr\_json.py
normalize\_scenarios.py
split\_normalized\_scenarios.py
kbcanonize.py

2. only with the argument 'train' or no argument (others dont need to be canonized), execute:

$ python3 canonize.py 




