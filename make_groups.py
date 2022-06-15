import os
import argparse
import pandas as pd
import numpy as np
import time
from more_itertools import set_partitions


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
    for i in range(len(df)):
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

    print("Advanced track: ", new_df.loc[new_df["adv_track"] == "Yes", "name"])
    # Remove people who are taking the advanced track (we will manually put them into groups)
    new_df = new_df.loc[new_df["adv_track"] != "Yes", :]

    new_df.reset_index(drop=True, inplace=True)
    # Convert the names in the work_with and avoid columns to indices
    name_idx_map = {k: v for k, v in zip(new_df["name"].values, new_df["name"].index)}
    new_df["work_with_idx"] = new_df["work_with"].map(name_idx_map)
    new_df["avoid_idx"] = new_df["avoid"].map(name_idx_map)
    return new_df

def get_overlaps_names(df, names):
    """Takes a list of names and finds the number of overlapping timeslots between them"""
    sub_df = df.loc[df["name"].isin(names), TIME_COLS]
    return (sub_df.sum() == len(names)).sum()

def is_le_k_overlaps(time_arr, k):
    """
    Check to see whether there are < k timeslot overlaps amongst the members in this array
    """
    # """Takes a numpy array and list of indices and finds the number of overlapping timeslots between them"""
    overlaps = (time_arr.sum(axis=0) == time_arr.shape[0]).sum()
    return overlaps < k

def num_overlaps(df, group):
    """Get the number of timeslot overlaps for this group"""
    time_arr = df.loc[group, TIME_COLS].to_numpy()
    return (time_arr.sum(axis=0) == time_arr.shape[0]).sum()

def valid_work_with(partition, work_with_arr):
    for group in partition:
        sub_arr = work_with_arr[group]
        idxs = np.unique(sub_arr[~np.isnan(sub_arr)])
        valid = np.all([int(idx) in group for idx in idxs])
        if not valid:
            return False
    return True

def valid_avoid(partition, avoid_arr):
    for group in partition:
        sub_arr = avoid_arr[group]
        idxs = np.unique(sub_arr[~np.isnan(sub_arr)])
        valid = ~np.any([int(idx) in group for idx in idxs])
        if not valid:
            return False
    return True

def valid_experience(partition, exp_arr):
    for group in partition:
        valid = max(exp_arr[group].sum(axis=0)) == len(group)
        if not valid:
            return False
    return True

def get_best_partition(df, group_size=3, k=3, num_trials=1000):
    """
    Creates the groups that compatible and have several timeslot overlaps between members
    1. Initialise le_k_overlaps = len(df) / 3
    2. Randomly partition group
    2. Check if subgroups are compatible (experience and personal conflicts)
    3. Count the number of groups in which there are < n timeslot overlaps.
    4. If the above is less than le_n_timeslow_overlaps, store partition and override var.
    5. Repeat 1-4.
    """

    le_k_overlaps = len(df) // group_size
    exp_arr = df[["level1", "level2"]].to_numpy()
    work_with_arr = df["work_with_idx"].to_numpy()
    avoid_arr = df["avoid_idx"].to_numpy()
    time_arr = df[TIME_COLS].to_numpy()
    n = [col for col in df if col not in TIME_COLS]

    optimal_partition = []
    idxs = list(range(len(df)))
    for i in range(num_trials):
        np.random.shuffle(idxs)
        partition = [idxs[i:i + group_size] for i in range(0, len(idxs), group_size)]
        if not valid_work_with(partition, work_with_arr):
            continue
        if not valid_experience(partition, exp_arr):
            continue
        if not valid_avoid(partition, avoid_arr):
            continue
        # Count the number of groups with < k overlaps
        partition_le_k_overlaps = 0
        for group in partition:
            partition_le_k_overlaps += is_le_k_overlaps(time_arr[group], k)

        if partition_le_k_overlaps < le_k_overlaps:
            le_k_overlaps = partition_le_k_overlaps
            optimal_partition = partition

    le_k_groups = [g for g in optimal_partition if not is_le_k_overlaps(time_arr[g], k)]
    ge_k_groups = [g for g in optimal_partition if g not in le_k_groups]
    return ge_k_groups, le_k_groups


def run(args):
    df = pd.read_csv(args.csv_file)
    # Only take the most recent entry by each person
    df.drop_duplicates(Q_NUMS["name"], keep="last", ignore_index=True, inplace=True)
    df.drop("Timestamp", axis=1, inplace=True)

    df = clean_df(df)

    # Get the total number of people available for each timeslot
    # slot_availability = df[TIME_COLS].sum()

    # Get total number of overlaps for each person (taking care to not count self overlaps)
    # df["TotalOverlaps"] = (df[TIME_COLS] * (slot_availability - df[TIME_COLS])).sum(axis=1)

    ge_k_groups, le_k_groups = get_best_partition(df, group_size=3, k=3, num_trials=1000)
    # TODO: Handle the le_k_groups, either by rearranging or by adding to existing groups
    print(ge_k_groups)
    print(le_k_groups)
    print("Groups with < k overlaps: ", le_k_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default = "data/example_CoworkingJun20-Jul3.csv")
    args = parser.parse_args()
    run(args)