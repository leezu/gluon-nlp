from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore


# Adapted from https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
class BoundedExecutor(ThreadPoolExecutor):
    """BoundedExecutor behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.
    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool
    """

    def __init__(self, bound, max_workers):
        self.semaphore = BoundedSemaphore(bound + max_workers)
        super(BoundedExecutor, self).__init__(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        """See concurrent.futures.Executor#submit"""
        self.semaphore.acquire()
        try:
            future = super(BoundedExecutor, self).submit(fn, *args, **kwargs)
        except:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future
