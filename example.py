import numpy as np
import faiss

# Example dataset: 1000 vectors of dimension 64
d = 64                           # Dimension of each vector
nb = 1000                        # Number of vectors in the database
np.random.seed(1234)             # Make the random numbers predictable
db_vectors = np.random.random((nb, d)).astype('float32')
db_vectors[:, 0] += np.arange(nb) / 1000. # Make vectors distinguishable

# Query: 10 vectors of dimension 64
nq = 10                          # Number of query vectors
query_vectors = np.random.random((nq, d)).astype('float32')
query_vectors[:, 0] += np.arange(nq) / 1000. # Make vectors distinguishable

# Building a flat (brute-force) index
index = faiss.IndexFlatL2(d)     # L2 distance for similarity
print('Index is trained:', index.is_trained)
index.add(db_vectors)            # Adding the database vectors
print('Index contains', index.ntotal, 'vectors')

# Searching for the 4 nearest neighbors
k = 4                            # Number of nearest neighbors to find
D, I = index.search(query_vectors, k) # D: Distances, I: Indices of neighbors

print('Nearest neighbors for the first 5 queries:\n', I[:5])
print('Distances:\n', D[:5])
