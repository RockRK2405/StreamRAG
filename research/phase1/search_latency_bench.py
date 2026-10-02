# Search-latency micro-benchmark on SYNTHETIC random-word corpora (latency only, says nothing about quality).
import time, statistics, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(20000)]
zipf = 1 / np.arange(1, len(vocab) + 1); zipf /= zipf.sum()
def bm25_index(docs, k1=1.5, b=0.75):
    cv = CountVectorizer(token_pattern=r"\S+"); tf = cv.fit_transform(docs).tocsc().astype(np.float32)
    N = tf.shape[0]; dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean(); df = np.diff(tf.indptr)
    idf = np.log(1 + (N - df + .5) / (df + .5)).astype(np.float32)
    tf = tf.tocoo(); denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
    w = sp.csc_matrix((idf[tf.col] * tf.data * (k1 + 1) / denom, (tf.row, tf.col)), shape=tf.shape)
    return cv, w
for n in (1_000, 10_000, 100_000):
    docs = [" ".join(rng.choice(vocab, 150, p=zipf)) for _ in range(n)]
    t = time.perf_counter(); cv, W = bm25_index(docs); build = time.perf_counter() - t
    qs = [" ".join(rng.choice(vocab[:3000], 6)) for _ in range(50)]
    lat = []
    for q in qs:
        t = time.perf_counter(); qv = cv.transform([q]); cols = qv.indices
        s = np.asarray(W[:, cols].sum(1)).ravel(); top = np.argpartition(-s, 50)[:50]; lat.append((time.perf_counter() - t) * 1000)
    E = rng.standard_normal((n, 384)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    dl = []
    for _ in range(50):
        qv = rng.standard_normal(384).astype(np.float32)
        t = time.perf_counter(); s = E @ qv; top = np.argpartition(-s, 50)[:50]; dl.append((time.perf_counter() - t) * 1000)
    print(f"n={n:>7}: bm25_build={build:.2f}s bm25_query_p50={statistics.median(lat):.2f}ms "
          f"dense384_bruteforce_p50={statistics.median(dl):.2f}ms dense_matrix={E.nbytes/1e6:.1f}MB")
