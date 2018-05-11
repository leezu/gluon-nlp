import numba
import numpy as np


@numba.extending.overload(np.unique)
def np_unique(a, return_counts=False):
    def np_unique_impl(a):
        b = np.sort(a.ravel())
        head = list(b[:1])
        tail = [x for i, x in enumerate(b[1:]) if b[i] != x]
        return np.array(head + tail)

    def np_unique_wcounts_impl(a):
        b = np.sort(a.flatten())
        unique = list(b[:1])
        counts = [1 for _ in unique]
        for x in b[1:]:
            if x != unique[-1]:
                unique.append(x)
                counts.append(1)
            else:
                counts[-1] += 1
        return unique, counts

    if not return_counts:
        return np_unique_impl
    else:
        return np_unique_wcounts_impl



@numba.njit(nogil=True)
def test(a):
    return np.unique(a)

def test2(a):
    return np.unique(a, return_counts=True)

a_ = np.array([0,0,1,2,3])
print(a_)
print(test(a_))
print(test2(a_))
