# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import mxnet as mx


class SparseDenseDivision(mx.operator.CustomOp):
    '''Sparse L2Normalization operator.
    '''

    def forward(self, is_train, req, in_data, out_data, aux):
        lhs = in_data[0]
        rhs = in_data[0]
        if lhs.stype == 'row_sparse':
            if rhs.stype == 'row_sparse':
                # TODO
                out_data = mx.nd.broadcast_div(lhs.data, rhs.data)
            else:
                out_data = mx.nd.broadcast_div(lhs.data, rhs)

            out = mx.nd.sparse.row_sparse_array((out_data, lhs.indices),
                                                shape=lhs.shape)
        else:
            out = mx.nd.broadcast_div(lhs, rhs)
        self.assign(out_data[0], req[0], out)


@mx.operator.register('dense_division')
class SparseDenseDivisionProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SparseDenseDivisionProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def infer_storage_type(self, in_stype):
        '''Infer storage type logic for the forward pass
        Takes a list of storage types for inputs
        Returns three lists lists, one for input storage types inferred,
        second for output storage types inferred and third for aux storage
        types inferred
        The in_stype is the list containing storage type for inputs
        If the input is a dense ndarray then we infer the input
        and output to be dense. If input is row_sparse then input and output
        are inferred as row_sparse.
        '''
        if in_stype[0] == 'default':
            return ['default'], ['default'], []
        return ['row_sparse'], ['row_sparse'], []

    def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype,
                                    igrad_stype, aux_stype):
        '''Infer storage type logic for the backward pass
        Takes storage type of output gradients(ograd_stype), inputs(in_stype),
        outputs(out_stype) and aux(aux_stype).
        Returns inferred storage types in the following order:
        ograd_stype, in_stype, out_stype, igrad_stype (Storage type for input gradients)
        and aux_stype.
        '''
        if in_stype[0] == 'default':
            return ['default'], ['default'], ['default'], ['default'], []
        return ['row_sparse'], ['row_sparse'], ['row_sparse'], ['default'], []

    def create_operator(self, ctx, shapes, dtypes):
        return SparseDenseDivision()


if __name__ == '__main__':
    #Default
    x = mx.nd.random_uniform(1, 10, shape=(4, 10))
    y = mx.nd.Custom(x, op_type='sparse_l2normalization')
    print("Original ndarray")
    print("--------------")
    print(x.asnumpy())
    print("Normalized ndarray")
    print("--------------")
    print(y.asnumpy())
    print("stype of input is {}".format(x.stype))
    print("stype of output is {}".format(y.stype))

    #Sparse
    x = x.tostype('row_sparse')
    y = mx.nd.Custom(x, op_type='sparse_l2normalization')
    print("Original ndarray")
    print("--------------")
    print(x.asnumpy())
    print("Normalized ndarray")
    print("--------------")
    print(y.asnumpy())
    print("stype of input is {}".format(x.stype))
    print("stype of output is {}".format(y.stype))
