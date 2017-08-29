# SearchEngine Class

def create_tm_dict(src_file, tm_file):
    tm_dict = {}
    with open(src_file) as f_src, open(tm_file) as f_tm:
        for src, tms in zip(f_src, f_tm):
            src = tuple(src.rstrip("\n"))
            tms = tms.rstrip("\n").split("\t")
            n_tms = len(tms)//3

            tm_src = tms[:n_tms]
            tm_trg = tms[n_tms:2*n_tms]
            tm_score = tms[2*n_tms:3*n_tms]

            tm_src = map(lambda x: x.split(" "), tm_src)
            tm_trg = map(lambda x: x.split(" "), tm_trg)
            tm_score = map(float, tm_score)

            tm_dict[src] = (tm_src, tm_trg, tm_score)

    return tm_dict

class SearchEngine:
    """Search Engine class (to load TM's)"""
    def __init__(self, src_file, tm_file):
        self.internal_dict = create_tm_dict(src_file, tm_file)
    
    def fss(self, src):
        """Queries search engine for src"""
        return self.internal_dict.get(src)