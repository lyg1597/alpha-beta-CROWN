import numpy as np 
import random 

def reorg_bounds(bounds, pivot):
    pivot_val = bounds[pivot]
    # res = [pivot_val]
    bounds_left = []
    bounds_right = []
    for i in range(len(bounds)):
        if i!=pivot:
            val = bounds[i]
            # A concrete less than or equal to
            if val[1] <= pivot_val[0]:
                bounds_left.append(val) 
            # A concrete greater than or equal to
            elif pivot_val[1] <= val[0]:
                bounds_right.append(val)
            # Compare based on center of interval
            elif (val[0]+val[1])/2 < (pivot_val[0]+pivot_val[1])/2:
                bounds_left.append(val)
            else:
                bounds_right.append(val)
    return bounds_left, bounds_right

def sort_bounds(bounds):
    if len(bounds) == 0:
        return bounds
    elif len(bounds) == 1:
        return bounds
    else:
        pivot = int(len(bounds)/2)
        bounds_left, bounds_right = reorg_bounds(bounds, pivot)
        sort_left = sort_bounds(bounds_left)
        sort_right = sort_bounds(bounds_right)
        return sort_left + [bounds[pivot]] + sort_right

def get_set_order(sorted_bounds):
    res_list = []
    for i in range(len(sorted_bounds)):
        bins = []
        ref_bound = sorted_bounds[i]
        for j in range(len(sorted_bounds)):
            bound = sorted_bounds[j]
            if ref_bound[0]<=bound[1]<=ref_bound[1] or \
                ref_bound[0]<=bound[0]<=ref_bound[1] or \
                bound[0]<=ref_bound[0]<=bound[1] or \
                bound[0]<=ref_bound[1]<=bound[1]:
                bins.append(bound[2])
        res_list.append(bins)
    return res_list

if __name__ == "__main__":
    N = 10
    tmp = np.random.uniform(0,5,(N,2))
    res_array = np.zeros(tmp.shape)
    res_array[:,0] = np.min(tmp, axis=1)
    res_array[:,1] = np.max(tmp, axis=1)
    res_array = np.hstack((res_array, np.arange(0,N).reshape((-1,1))))
    print(res_array)
    sorted_bounds = sort_bounds(res_array)
    print(sorted_bounds) 
    set_order = get_set_order(sorted_bounds)
    print(set_order)