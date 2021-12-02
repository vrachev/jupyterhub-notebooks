import requests
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from more_itertools import pairwise

from nblib import data


@dataclass
class Info:
    client: data.ClientConnection
    max_tasks: int
    max_tests: int
    batch: int
    build_a: str
    build_a_label: str
    build_b: str
    build_b_label: str
    evg_base: str = "https://cedar.mongodb.com/rest/v1/perf/version"


class Analysis:
    """Class containing functions for the perf discovery notebook."""

    def __init__(self, pd_info: Info):
        self.pd_info = pd_info

    def read_task_list(self, build):
        """Read the list of tasks for the a build from the Cedar REST api."""

        print(f"Fetching tasks for {build}")
        fetched = 0
        data = {
            "project": [],
            "variant": [],
            "task": [],
            "test": [],
            "measurement": [],
            "args": [],
            "execution": [],
            "value": [],
        }

        while True:
            r = requests.get(f"{self.pd_info.evg_base}/{build}?skip={fetched}&limit={self.pd_info.batch}")
            if r.status_code == 404:
                raise r.raise_for_status()

            for cp in r.json():
                if cp["rollups"]["stats"]:
                    for rollup in cp["rollups"]["stats"]:
                        data["project"].append(cp["info"]["project"])
                        data["variant"].append(cp["info"]["variant"])
                        data["task"].append(cp["info"]["task_name"])
                        data["test"].append(cp["info"]["test_name"])
                        data["measurement"].append(rollup["name"])
                        data["args"].append(cp["info"]["args"])
                        data["execution"].append(cp["info"]["execution"])
                        data["value"].append(rollup["val"])
            print(f"Tasks fetched: {fetched}", end='\r')
            if (fetched + self.pd_info.batch) > self.pd_info.max_tasks:
                # because of carriage return above, print this one last time so it doesn't get overwritten.
                print(f"Tasks fetched: {fetched}")
                print(f"Finished fetching tasks")
                break
            fetched += self.pd_info.batch

        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def filter_and_merge(dfa: pd.DataFrame, dfb: pd.DataFrame):
        """Filter and merge the tasks from the 2 commits"""

        # TODO: Make filters user configurable
        def filter_canaries(df):
            df_filtered = df[~df.test.str.match(
                'CleanUp|canary|fio|iperf|NetworkBandwidth|finishing|Setup|Quiesce|GennyOverhead')]
            df_filtered = df_filtered[~df_filtered.test.str.contains('ActorFinished|ActorStarted|Setup')]
            return df_filtered

        print(f"dfa length = {len(dfa)}, dfb length = {len(dfb)}")

        dfa = filter_canaries(dfa)
        dfb = filter_canaries(dfb)

        print(f"filtered dfa length = {len(dfa)}, filtered dfb length = {len(dfb)}")

        dfa["args"] = dfa["args"].apply(json.dumps)
        dfb["args"] = dfb["args"].apply(json.dumps)

        # merge our results together:
        comparison = dfa.merge(dfb, on=["project", "variant", "task", "test", "measurement", "args"])

        print(f"length of merged comparison = {len(comparison)}")

        found_ts = comparison[["project", "variant", "task", "test", "measurement", "args"]]

        # We drop duplicates since there could be multiple executions for the same combination of the properties below.
        found_ts = found_ts.drop_duplicates()

        print(f"length after de-dup = {len(found_ts)}")

        # keep the interesting metrics
        found_ts = found_ts[found_ts["measurement"].isin(['AverageLatency',
                                                          'ops_per_sec',
                                                          'system cpu user (%) - mean',
                                                          'ss mem resident (MiB) - mean',
                                                          'Data - disk xvde utilization (%) - mean',
                                                          'Journal - disk xvdf utilization (%) - mean'])]

        print(f"length after keeping interesting metrics = {len(found_ts)}")

        return comparison, found_ts

    @staticmethod
    def get_stable_region(commit_date, ts, cps):
        """Algorithm to look up time series in order to characterize the stable region of results around build_a."""

        true_positive_orders = {
            cp["order"]
            for cp in cps
            if cp["triage"]["triage_status"] == "true_positive"
        }
        len_ts = len(ts["data"])
        stable_region_bounds = (
                [0]
                + [idx for idx, datum in enumerate(ts["data"]) if datum["order"] in true_positive_orders]
                + [len_ts]
        )

        start = end = 0

        # if base commit before or after the entire time series, get the closest stable region
        if commit_date < ts["data"][0]["commit_date"]:
            # first stable region
            start = stable_region_bounds[0]
            end = stable_region_bounds[1]

        if commit_date > ts["data"][len_ts - 1]["commit_date"]:
            # last stable region
            start = stable_region_bounds[-2]
            end = stable_region_bounds[-1]

        for start_bound, end_bound in pairwise(stable_region_bounds):
            if (
                    ts["data"][start_bound]["commit_date"]
                    <= commit_date
                    <= ts["data"][end_bound - 1]["commit_date"]
            ):
                start = start_bound
                end = end_bound
        return [datum["value"] for datum in ts["data"][start:end]]

    def process_stable_regions(self, found_ts):
        """Calculate the means and std dev for the Zscores"""
        # limit number of tests
        found_ts = found_ts[0:self.pd_info.max_tests]

        total = len(found_ts)

        stable_mean = []
        stable_std = []
        stable_length = []

        date_a = self.pd_info.client["expanded_metrics"]["versions"].find_one({"version_id": self.pd_info.build_a})["commit_date"]
        date_b = self.pd_info.client["expanded_metrics"]["versions"].find_one({"version_id": self.pd_info.build_b})["commit_date"]

        for index, row in found_ts.iterrows():
            # some tests do not have threads.
            if row["args"] == "null":
                row["args"] = "{}"
            ts = self.pd_info.client["expanded_metrics"]["time_series"].find_one({
                "project": row["project"],
                "variant": row["variant"],
                "task": row["task"],
                "test": row["test"],
                "args": json.loads(row["args"]),
                "measurement": row["measurement"],
            })
            cps = list(self.pd_info.client["expanded_metrics"]["change_points"].find({
                "time_series_info.project": row["project"],
                "time_series_info.variant": row["variant"],
                "time_series_info.task": row["task"],
                "time_series_info.test": row["test"],
                "time_series_info.args": json.loads(row["args"]),
                "time_series_info.measurement": row["measurement"],
            }))

            try:
                stable_region = self.get_stable_region(date_a, ts, cps)
                stable_mean.append(np.mean(stable_region))
                stable_std.append(np.std(stable_region))
                stable_length.append(len(stable_region))
            except Exception as e:
                # no stable region found
                print("")
                print(f"no stable region found for {len(stable_length)}")
                print("")
                stable_mean.append(np.nan)
                stable_std.append(np.nan)
                stable_length.append(0)
                pass

            print(f"Stable regions processed: {len(stable_length)}/{total}", end="\r")

        print('')
        found_ts.insert(0, "stable_mean", stable_mean)
        found_ts.insert(1, "stable_std", stable_std)
        found_ts.insert(2, "stable_length", stable_length)

        return found_ts, date_a, date_b
