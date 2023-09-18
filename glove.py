import numpy as np
import time


start = time.time()
word2idx = dict()  # map words to vector
idx2word = []  # map vector back to word
# Download vectors from here: https://nlp.stanford.edu/projects/glove/
embeddings = 'glove.6B.50d.txt'
with open(f'C:/Users/aroym/Downloads/glove.6B/{embeddings}',
          'r',
          encoding='utf-8') as f:
    vec_list = []
    try:
        for line in f:
            components = line.split()
            word = components[0]
            vec_list.append(np.asarray(components[1:], "float32"))
            word2idx[word] = len(vec_list) - 1
            idx2word.append(word)
    except:
        print(f'Hit exception when reading {line}')
    vectors = np.stack(vec_list, axis=0)
print(f'Loaded {embeddings} vectors in {time.time() - start} seconds')
king = vectors[word2idx['king']]
queen = vectors[word2idx['queen']]
man = vectors[word2idx['man']]
woman = vectors[word2idx['woman']]
query = king + (woman - man)
query = np.expand_dims(query, 0)
# Squared L2 dist.
diff = vectors - query
diff = np.sum(diff * diff, axis=1)
closest = np.argsort(diff)
print('vec(king) +  (vec(woman) - vec(man)) = ? (L2 distance)')
for i in range(10):
    print(f'{idx2word[closest[i]]}')
