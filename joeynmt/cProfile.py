# script to invoke joeynmt not as module 
from joeynmt.training import train
from joeynmt.prediction import test
from joeynmt.prediction import translate

def main(config_path):

    train(cfg_file=config_path)

if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    assert len(argv) == 1, argv # just  need 'configs/kvr/mycfg.yaml" for this script
    main()
