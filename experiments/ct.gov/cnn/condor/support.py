import numpy as np


def subset_generator(container):
    """Yield random subset of label names
    
    Parameters
    ----------
    container : list of label names to yield
    
    Generate a random bit string and yield labels which correspond to
    those bit strings. Make sure not to yield an empty label set!
    
    """
    container = np.array([str(elem) for elem in container])

    while True:
        bit_string = np.random.randint(low=2, size=len(container))

        subset = container[bit_string==0]
        if subset.size:
            yield ','.join(subset)
    
def int_generator(lo, hi):
    """Generate random integer in the range [lo,hi]"""
    
    while True:
        yield np.random.randint(low=lo, high=hi+1)

def pow_generator(lo, hi, pow=2):
    """Generate random integer in the range [pow**lo, pow**hi]"""
    
    while True:
        yield pow ** np.random.randint(low=lo, high=hi+1)

def float_generator(lo, hi, pow=10):
    """Generate random floats in the range [lo, hi]"""
    
    while True:
        num = (hi-lo)*np.random.random_sample() + lo
        
        yield pow**num

def const_generator(elem):
    """Generates the element ad infinitum"""

    while True:
        yield elem

def item_generator(container):
    """Generates one element from the container"""

    while True:
        yield np.random.choice(container)

def my_range(start, end):
    for i in range(start, end):
        yield i

def cycle_generator(collection):
    """Yield items in order ad infinitum"""

    while True:
        for elem in collection:
            yield elem


def all_subsets_generator(label_names):
    """Yield all subsets of labels"""

    while True:
        for bit_str in range(1, 256):
            mask = np.array([int(bit) for bit in '{0:08b}'.format(bit_str)], dtype=np.bool)
            
            yield ','.join(label_names[mask])


if __name__ == '__main__':
    label = subset_generator(labels)

    for _ in range(10):
        print next(label)
