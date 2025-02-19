
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
