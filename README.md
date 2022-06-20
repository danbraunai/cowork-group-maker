# cowork-group-maker
Organises people into groups based on preferences such as available times.

The process to create groups is as follows:
1. Save the csv file output of the google form in data/ (currently expects the form used in MLSS).
2. If you want to manually form groups, add them to data/ and set the --manual_group_file flag.
3. run python make_groups.py with the --csv_in flag pointing to the input csv. This will produce
    an output file (in the location specified by the --csv_out flag).