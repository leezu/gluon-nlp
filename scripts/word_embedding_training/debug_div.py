import mxnet as mx

a = mx.nd.sparse.row_sparse_array(
    (mx.nd.array([[1, 2, 3], [1, 2, 3]]), mx.nd.arange(2, dtype=int)),
    ctx=mx.gpu())
b = mx.nd.sparse.row_sparse_array((mx.nd.arange(2).reshape(
    (-1, 1)) + 1, mx.nd.arange(2, dtype=int)), ctx=mx.gpu())

print(a.data)
print(b.data)
print(mx.nd.sparse.dense_division(a, b).data)
