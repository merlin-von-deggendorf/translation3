import threading


# comparing two sequences with the needleman-wunsch algorithm (global alignment)
# generate custom implementation of the needleman-wunsch algorithm
def NeedlemanWunsch(seq1,seq2):
    # initialize the matrix
    matrix = [[0 for i in range(len(seq1)+1)] for j in range(len(seq2)+1)]
    # initialize the first row
    for i in range(1,len(seq1)+1):
        matrix[0][i] = i
    # initialize the first column
    for i in range(1,len(seq2)+1):
        matrix[i][0] = i
    # fill the matrix
    for i in range(1,len(seq2)+1):
        for j in range(1,len(seq1)+1):
            if seq1[j-1] == seq2[i-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1],matrix[i-1][j],matrix[i][j-1])+1
    # return the distance
    return matrix[len(seq2)][len(seq1)]

def NeedlemanWunschRatio(seq1,seq2):
    return 1-NeedlemanWunsch(seq1,seq2)/max(len(seq1),len(seq2))

class ParallelNeedleManWunschRatio:
    def __init__(self,maxThreads):
        self.maxThreads = maxThreads
        self.semaphore = threading.Semaphore(maxThreads)
        self.lock = threading.Lock()
    def worker(self, seq1, seq2, on_finish):
        try:
            # Calculate the result using NeedlemanWunschRatio
            result = NeedlemanWunschRatio(seq1, seq2)
            # Store the result in a thread-safe way
            with self.lock:
                on_finish(result)
        finally:
            # Make sure to release the semaphore so another thread can start
            self.semaphore.release()

    def calculate(self, seq1, seq2,on_finish):
        # Wait until a thread slot is available
        self.semaphore.acquire()
        # Start a new thread for the calculation
        t = threading.Thread(target=self.worker, args=(seq1, seq2, on_finish))
        t.start()

print(NeedlemanWunschRatio("hellods","helooodsasdfasdfasdf"))