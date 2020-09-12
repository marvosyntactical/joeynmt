
def split_string_if_match_from_delimiter_set(s,delimiter_set):
    x,y = None, None
    for delimiter in delimiter_set:
        include_delim = len(delimiter)  
        match = s.find(delimiter)
        if match != -1:
            split_idx = match + include_delim
            x, y = s[:split_idx],s[split_idx:]
            break
    if x is None:
        assert False, s
    print(x," :::: \n :::: ",y)

    return x,y



def process(path, file_, in_file_ext,splitpart,EXT,src_trg_prefix="",split_char=None, split_attrs=None):

    with open(path+file_+in_file_ext, "r") as infile:
        line_list = infile.readlines()

    dataset = [line[2:] for line in line_list if line] # remove numbering and empty lines

    if split_char is not None:
        # split dataset into two on splitchar
        dataset = [line.split(split_char) for line in line_list]
    else:
        # split dataset into two upto list of given 'split_attrs'
        assert split_attrs is not None, f"need to split on either a delimiter or set of attributes"
        dataset = [split_string_if_match_from_delimiter_set(line,split_attrs) for line in dataset if line]

    dataset = [(x+"\n", y) for x, y in dataset]
    dataset = list(zip(*dataset))

    src_tgt = [src_trg_prefix+suffix for suffix in ["src", "trg"]]

    for i, part in enumerate(src_tgt):
        outfile = path+splitpart+"."+part+EXT
        input(outfile)

        with open(outfile, "w") as outfile:
            outfile.writelines(dataset[i])


def main(args):

    EXT = "FINAL"

    splitpart = "dev"
    if args != 0:
        splitpart = args[1]

    ext = ".txt"
    movie_dir = "../movie1_qa/"
    file_stem = "task1_qa_"
    file_ = file_stem+splitpart

    process(movie_dir,file_, ext, splitpart, EXT, split_char="\t")

    split_attrs = [
        "written_by",
        "starred_actors",
        "release_year",
        "has_genre",
        "has_plot",
        "directed_by",
        "has_tags",
        "in_language",
        "has_imdb_votes",
        "has_imdb_rating",
    ]
    kb_file_name = "movie_kb"
    process(movie_dir, kb_file_name, ext, splitpart, EXT, src_trg_prefix="kb", split_attrs=split_attrs)






    return 0


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if argv:
        main(argv)
    else:
        main(0)
