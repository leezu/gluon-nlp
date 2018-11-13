"""Python concurrent.futures executors
======================================

This file contains a lazy ThreadPoolExecutor. The ThreadPoolExecutor in Python
standard library first fetches the complete iterable, before using a thread
pool to apply the transformation. This is a major problem for us, as we must
load all data to memory but need to iterate lazily.

"""

import collections
import itertools
import time
from concurrent.futures import (Executor, ThreadPoolExecutor,
                                ProcessPoolExecutor)
from concurrent.futures.process import _process_chunk, _get_chunks
from functools import partial


class LazyExecutor(Executor):
    """ThreadPoolExecutor with lazy iterable collection in map().

    Implmentation taken from https://github.com/python/cpython/pull/707

    """

    def map(self, fn, *iterables, timeout=None, chunksize=1, prefetch=None):
        # pylint: disable=arguments-differ
        """Lazy apdaption of ThreadPoolExecutor.map.

        Unlike ThreadPoolExecutor.map:
        - iterables are prefetched lazily
        - if only a single iterable is specified, iter(iterables[0]) is used
          instead of zip(*iterables) to obtain a iterator over the arguments
          that are mapped to fn. This is to match the behavior of
          mxnet.gluon.Dataset.transform and gluonnlp.data.DataStream.transform
          which unpack argument tuples.

        """
        if timeout is not None:
            end_time = timeout + time.time()
        if prefetch is None:
            prefetch = self._max_workers
        if prefetch < 0:
            raise ValueError('prefetch count may not be negative')

        if len(iterables) > 1:
            argsiter = zip(*iterables)
        else:
            argsiter = iter(iterables[0])
        fs = collections.deque(
            self.submit(fn, *args)
            for args in itertools.islice(argsiter, self._max_workers +
                                         prefetch))

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def _result_iterator():
            nonlocal argsiter
            try:
                while fs:
                    res = fs[0].result() if timeout is None else fs[0].result(
                        end_time - time.time())
                    # Got a result, future needn't be cancelled
                    del fs[0]
                    # Dispatch next task before yielding to keep pipeline full
                    if argsiter:
                        try:
                            args = next(argsiter)
                        except StopIteration:
                            argsiter = None
                        else:
                            fs.append(self.submit(fn, *args))
                    yield res
            finally:
                for future in fs:
                    future.cancel()

        return _result_iterator()


class LazyThreadPoolExecutor(LazyExecutor, ThreadPoolExecutor):
    """ThreadPoolExecutor with lazy iterable collection in map().

    Implmentation taken from https://github.com/python/cpython/pull/707

    """

    pass


class LazyProcessPoolExecutor(LazyExecutor, ProcessPoolExecutor):
    """ProcessPoolExecutor with lazy iterable collection in map().

    Implmentation taken from https://github.com/python/cpython/pull/707

    """

    def map(self, fn, *iterables, timeout=None, chunksize=1, prefetch=None):
        """Returns an iterator equivalent to map(fn, iter).
        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(
            partial(_process_chunk, fn),
            _get_chunks(*iterables, chunksize=chunksize), timeout=timeout,
            prefetch=prefetch)
        return itertools.chain.from_iterable(results)
