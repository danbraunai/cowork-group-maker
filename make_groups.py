import os
import argparse
import pandas as pd
import numpy as np
import time
from more_itertools import set_partitions
from utils import *


# Set the mapping from questions to their given numbers in the csv
Q_NUMS = {
    "name": "Q1",
    "exp": "Q2",
    "availability": "Q3",
    "work_with": "Q4",
    "avoid": "Q5",
    "adv_track": "Q6"
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_COLS = [f"{Q_NUMS['availability']} [{day}]" for day in DAYS]
TIMESLOTS = [
    "12am-2am", "2am-4am", "4am-6am", "6am-8am", "8am-10am", "10am-12pm",
    "12pm-2pm", "2pm-4pm", "4pm-6pm", "6pm-8pm", "8pm-10pm", "10pm-12am",
]
TIME_COLS = [f"{day}_{time}" for day in DAYS for time in TIMESLOTS]


def clean_df(df):
    """
    Create a column for every time slot with a boolean value indicating a person's
    availability.

    df: Pandas dataframe with columns Q1 (the name) and all the columns in DAY_COLS (among others).

    Returns:
        new_df: df with columns equal to the keys of Q_NUMS except that availability is split
            into a boolean column for each timeslot in TIME_COLS
    """
    new_data = []
    for i in df.index:
        # Set the data for columns other than availability
        person = {k: df.loc[i, v] for k, v in Q_NUMS.items() if k not in ["availability", "exp"]}
        # Set level1==True if person answered 1 or 2 to Q2, and level2=True if entered 2 or 3.
        person["level1"] = df.loc[i, Q_NUMS["exp"]] in [1, 2]
        person["level2"] = df.loc[i, Q_NUMS["exp"]] in [2, 3]
        # Initially set all availability to false
        person.update({k: False for k in TIME_COLS})
        for col in DAY_COLS:
            # Get the day (from between the square brackets)
            day = col.split("[")[-1].split("]")[0]
            try:
                slots = df.loc[i, col].split(";")
            except AttributeError:
                # Person has no availability for this day
                pass
            else:
                for slot in slots:
                    person[f"{day}_{slot}"] = True

        new_data.append(person)
    new_df = pd.DataFrame(new_data)

    print("Advanced track: ", new_df.loc[new_df["adv_track"] == "Yes", "name"].tolist())

    new_df.reset_index(drop=True, inplace=True)
    return new_df

def match_requested_groups(df, group_size=3, k=3):
    """
    Creates groups for those who have requested to work with others.
    1. If someone requested a person that didn't fill out a form, ignore this request
    2. Verify that there are no conflicts (e.g. A wants to work with B but B does not want to work
        with A.) and put requested people in pairs.
        - TODO: If A requests B but B requests C, from a triple. If any more complex, raise error.
    3. For the people who aren't in pairs/groups, order them by least to most overlaps with others
    4. For each pair/group, go down the list in 2 and extend the group with that person if there
        are >= k timeslot overlaps with the existing group.
    5. return the formed groups.
    """
    pairs = list(df.loc[~df["work_with_idx"].isnull(), "work_with_idx"].astype(int).items())
    # Check that a person doesn't exist in multiple pairs
    idxs = flatten(pairs)
    assert len(set(idxs)) == len(idxs), (
        "A person has been requested multiple times or have requested someone else"
    )
    # Check that a pair has >=k timeslot overlaps and doesn't contain a avoid conflict
    for pair in pairs:
        overlaps = get_overlaps_idx(df, pair, TIME_COLS)
        assert overlaps >= k, (
            f"Pair {pair} only has {overlaps} timeslot overlaps"
        )
        for person in pair:
            assert df.loc[person, "avoid_idx"].astype(int) not in pair, (
                f"{df.loc[person, 'name']} does not want to work in {pair}"
            )

    # Order leftovers by least to most overlaps with others
    leftovers = df.loc[~df.index.isin(idxs)].copy()
    # Get total number of overlaps for each person (taking care to not count self overlaps)
    leftovers["TotalOverlaps"] = (
        leftovers[TIME_COLS] * (leftovers[TIME_COLS].sum() - leftovers[TIME_COLS])
    ).sum(axis=1)
    leftovers.sort_values(by="TotalOverlaps", ascending=True, ignore_index=False, inplace=True)

    # Find people to fill out these groups
    groups = []
    for pair in pairs:
        group = list(pair)
        for idx in leftovers.index:
            if get_overlaps_idx(df, group + [idx], TIME_COLS) >= 3:
                group = group + [idx]
                if len(group) == group_size:
                    break
        if len(group) == group_size:
            groups.append(group)
    all_group_members = flatten(groups)
    leftover_idxs = [i for i in leftovers.index if i not in all_group_members]
    return groups, leftover_idxs

def get_best_partition(df, group_size=3, k=3, num_trials=1000):
    """
    Creates the groups that compatible and have several timeslot overlaps between members
    1. Initialise le_k_overlaps = len(df) / 3
    2. Randomly partition group
    2. Check if subgroups are compatible (experience and personal conflicts)
    3. Count the number of groups in which there are < n timeslot overlaps.
    4. If the above is less than le_n_timeslow_overlaps, store partition.
    5. Repeat 1-4.
    """

    le_k_overlaps = len(df) // group_size
    exp_arr = df[["level1", "level2"]].to_numpy()
    avoid_arr = df["avoid_idx"].to_numpy()
    time_arr = df[TIME_COLS].to_numpy()

    optimal_partition = []
    # Note that df.index will not be of the form range(0,len(df)) due to previous operations.
    # We assume that it is, and then map back to the original idxs before returning
    idxs = list(range(len(df)))
    for i in range(num_trials):
        valid = True
        np.random.shuffle(idxs)
        partition = [idxs[i:i + group_size] for i in range(0, len(idxs), group_size)]
        if i % (num_trials // 5) == (num_trials // 5 - 1):
            print(f"{i + 1} - le k overlaps: {le_k_overlaps}")
        for g in partition:
            if not valid_exp(g, exp_arr) or not valid_avoid(g, avoid_arr):
                valid = False
                break
        if not valid:
            continue
        for g in partition:
            if not valid_avoid(g, avoid_arr):
                continue
        # Count the number of groups with < k overlaps
        partition_le_k_overlaps = 0
        for g in partition:
            partition_le_k_overlaps += is_le_k_overlaps(time_arr[g], k)

        if partition_le_k_overlaps < le_k_overlaps:
            le_k_overlaps = partition_le_k_overlaps
            optimal_partition = partition

    if not optimal_partition:
        return [], list(df.index)
    leftover_idxs = flatten([list(df.index[g]) for g in optimal_partition if is_le_k_overlaps(time_arr[g], k)])
    # Add any groups of size <group_size to leftovers
    leftover_idxs += flatten([list(df.index[g]) for g in optimal_partition if len(g) != group_size])
    ge_k_groups = [
        df.loc[df.index[g], "name"].tolist()
        for g in optimal_partition if len(g) == group_size and not is_le_k_overlaps(time_arr[g], k)
    ]
    return ge_k_groups, leftover_idxs

def add_leftovers(df, groups, leftover_idxs, group_size=3):

    exp_arr = df[["level1", "level2"]].to_numpy()
    avoid_arr = df["avoid_idx"].to_numpy()
    no_overlappers = []
    for idx in leftover_idxs:
        most_overlaps = 0
        most_overlap_group_idx = None
        for i, g in enumerate(groups):
            if len(g) > group_size:
                continue
            group_idxs = df.loc[df["name"].isin(g)].index.tolist()
            new_g = group_idxs + [idx]
            overlaps = get_overlaps_idx(df, new_g, TIME_COLS)
            if overlaps > most_overlaps and valid_exp(new_g, exp_arr) and valid_avoid(new_g, avoid_arr):
                most_overlaps = overlaps
                most_overlap_group_idx = i
        if most_overlaps == 0:
            print(f"{df.loc[idx, 'name']} HAS NO OVERLAPS WITH ANY GROUP")
            no_overlappers.append(df.loc[idx, 'name'])
        else:
            groups[most_overlap_group_idx].append(df.loc[idx, "name"])
    return groups, no_overlappers

def run(args):
    raw_df = pd.read_csv(args.csv_in)
    # Only take the most recent entry by each person
    raw_df.drop_duplicates(Q_NUMS["name"], keep="last", ignore_index=True, inplace=True)
    raw_df.drop("Timestamp", axis=1, inplace=True)

    raw_df = clean_df(raw_df)

    if args.manual_group_file:
        manual_group_df = pd.read_csv(args.manual_group_file, header=None)
        manual_groups = [manual_group_df[col].dropna().tolist() for col in manual_group_df]
    else:
        manual_groups = []

    # Remove the people who are in manual groups
    df = raw_df.loc[~raw_df["name"].isin(flatten(manual_groups))].copy()
    df.reset_index(drop=True, inplace=True)

    # Convert the names in the work_with and avoid columns to indices
    name_idx_map = {k: v for k, v in zip(df["name"].values, df["name"].index)}
    df["work_with_idx"] = df["work_with"].map(name_idx_map)
    df["avoid_idx"] = df["avoid"].map(name_idx_map)

    # Create groups for the people who have requested to work with others
    requested_groups, leftover_idxs = match_requested_groups(df, group_size=3, k=3)
    requested_groups = [df.loc[g, "name"].tolist() for g in requested_groups]

    sub_df = df.loc[leftover_idxs]

    ge_k_groups, leftover_idxs = get_best_partition(
        sub_df, group_size=3, k=3, num_trials=args.num_trials
    )
    assert ge_k_groups, "No groups found, try increasing num_trials"
    # Run again just on the leftovers
    ge_k_groups2, leftover_idxs = get_best_partition(
        sub_df.loc[leftover_idxs], group_size=3, k=3, num_trials=args.num_trials // 10
    )

    all_groups = manual_groups + requested_groups + ge_k_groups + ge_k_groups2
    final_groups, no_overlappers = add_leftovers(df, all_groups, leftover_idxs)
    assert len(flatten(final_groups)) + len(no_overlappers) == len(raw_df), (
        "Some people are not assigned to groups nor are in the no_overlappers"
    )
    print("final_groups:", final_groups)
    print("no_overlappers:", no_overlappers)

    summary = []
    # Get the overlapping times for each group (and leftover individual)
    for group in final_groups + [[p] for p in no_overlappers]:
        members = "__".join(group)
        experience = get_group_experience(raw_df, group)
        timeslots = get_group_timeslots(raw_df, group, TIME_COLS)
        summary.append([members, experience, timeslots])

    out_df = pd.DataFrame(summary, columns=["Members", "Experience", "Timeslots"])
    out_df.to_csv(args.csv_out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", type=str, default="data/example_CoworkingJun20-Jul3.csv")
    parser.add_argument("--csv_out", type=str, default="data/example_GroupsJun20-Jul3.csv")
    parser.add_argument("--manual_group_file", type=str)
    parser.add_argument("--num_trials", type=int, default=1000000)
    args = parser.parse_args()
    run(args)