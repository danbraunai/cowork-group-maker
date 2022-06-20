import numpy as np


def get_overlaps_idx(df, idxs, cols):
    """Takes a list of idxs and finds the number of overlapping timeslots between them"""
    return (df.loc[idxs, cols].sum() == len(idxs)).sum()

def get_overlaps_names(df, names, cols):
    """Takes a list of idxs and finds the number of overlapping timeslots between them"""
    return (df.loc[df["name"].isin(names), cols].sum() == len(names)).sum()

def get_group_timeslots(df, names, cols):
    mask = df.loc[df["name"].isin(names), cols].sum() == len(names)
    timeslots = ",".join(mask[mask].index.tolist())
    return timeslots

def get_group_experience(df, names):
    mask = df.loc[df["name"].isin(names), ["level1", "level2"]].sum() == len(names)
    experience = ",".join(mask[mask].index.tolist())
    return experience

def is_le_k_overlaps(time_arr, k):
    """
    Check to see whether there are < k timeslot overlaps amongst the members in this array
    """
    # """Takes a numpy array and list of indices and finds the number of overlapping timeslots between them"""
    overlaps = (time_arr.sum(axis=0) == time_arr.shape[0]).sum()
    return overlaps < k

def num_overlaps(df, group, cols):
    """Get the number of timeslot overlaps for this group"""
    time_arr = df.loc[group, cols].to_numpy()
    return (time_arr.sum(axis=0) == time_arr.shape[0]).sum()

def valid_work_with(group, work_with_arr):
    """Test whether a group contains those who requested to work together."""
    sub_arr = work_with_arr[group]
    idxs = np.unique(sub_arr[~np.isnan(sub_arr)])
    return np.all([int(idx) in group for idx in idxs])

def valid_avoid(group, avoid_arr):
    """Test whether a group does not have any 'avoid' conflicts."""
    sub_arr = avoid_arr[group]
    idxs = np.unique(sub_arr[~np.isnan(sub_arr)])
    return ~np.any([int(idx) in group for idx in idxs])

def valid_exp(group, exp_arr):
    """Test whether a group does not have any 'experience' conflicts."""
    return max(exp_arr[group].sum(axis=0)) == len(group)

def flatten(l):
    return [el for sub in l for el in sub]