import torch

class SparseBooklet:

    """A set of preloaded reservoir weights matching the reservoir seed and approximate sparcity (if applicable)

    Parameters
    ----------
    book: dict
        the set of approximate reservoir tensors and associated connectivity thresholds
    keys:
        the key to access the dictionary (typically an approximate sparcity threshold)
    """
    def __init__(self, book, keys):
        self.sparse_book = book
        self.sparse_keys_ = np.array(keys)

    def get_approx_preRes(self, connectivity_threshold):
        """ Given a connectivity threshold, the method will return the sparse matrix most closely matching that threshold.

        Parameters
        ----------
        connectivity_threshold: float
            #TODO description

        """
        #print("sparse_keys", self.sparse_keys_, "connectivity_threshold", connectivity_threshold   )
        key_ =  self.sparse_keys_[self.sparse_keys_ > connectivity_threshold][0]
        val =  self.sparse_book[key_].clone()
        return val



class GlobalSparseLibrary:
    """
    This will approximate the search for the sparcity hyper-parameter, which will dramatically speed up training of the network.

    Parameters
    ----------
    lb: int
        lower bound (connectivity)
    ub: int
        upper bound (connectivity)
    n_nodes: number of nodes in the reservoir
    precision: the precision of the approximate sparcity metric
    flip_the_script: bool
        completely randomizes which reservoir has been selected.
    """

    def __init__(self, device, lb = -5, ub = 0, n_nodes = 1000, precision = None, 
                 flip_the_script = False):
        self.lb = lb
        self.ub = ub
        self.n_nodes_ = n_nodes
        self.library = {}
        self.book_indices = []
        self.precision = precision
        self.flip_the_script = flip_the_script
        self.device = device
        

    def addBook(self, random_seed):
        """
        Add a sparse reservoir set by looping through different different connectivity values and assigining one reservoir weight matrix per connetivity level
        and storing these for downstream use by RcNetwork
        We generate the reservoir weights and store them in the sparse library.

        Parameters
        ----------
        random_seed: the random seed of the SparseLibrary with which to make the preloaded reservoir matrices
        """

        book = {}
        n = self.n_nodes_
        
        random_state = torch.Generator(device = self.device).manual_seed(random_seed)

        accept = torch.rand(n, n, generator = random_state, device = self.device) 
        reservoir_pre_weights = torch.rand(n, n, generator = random_state, device = self.device) * 2 -1

        "for now we're going to avoid sparse matrices"
        for connectivity in np.logspace(self.ub, self.lb, self.precision): #, device = self.device):
            #book[connectivity] = csc_matrix((accept < connectivity ) * reservoir_pre_weights)
            
            book[connectivity] = (accept < connectivity ) * reservoir_pre_weights
        sparse_keys_ = sorted(book)

        self.library[random_seed] = SparseBooklet(book = book, keys = sparse_keys_)
        self.book_indices.append(random_seed)

    def getIndices(self):
        """returns book indices"""
        return self.book_indices

    def get_approx_preRes(self, connectivity_threshold, index = 0):
        """ This function is for use by RcNetwork to access different sets of reservoir matrices.
        Given a connectivity threshold we access a reservoir by approximate sparcity / connectivity.
        But which randomly generated reservoir we select is determined by the index, which is what ESN uses if the one reservoir is nilpotent.
        Parameters
        ----------
            connectivity threshold: float
            index: int
                which preloaded reservoir do we want to load? Each index references a difference Sparse Booklet (ie a different reservoir)

        Returns
        -------
        sparse booklet for reading downstream by RcNetwork class
        (we are returning a set of pre-loaded matrices to speed up optimization of the echo-state network by avoiding 
        repeated tensor generation.)
        """
        if self.flip_the_script:
            index = np.random.randint(len(self.book_indices))
        #print("index", index, "book indices", self.book_indices, "self.library", self.library)
        book = self.library[self.book_indices[index]]
        if index != 0:
            printc("retrieving book from library" + str(self.book_indices[index]), 'green')
        return book.get_approx_preRes(connectivity_threshold)