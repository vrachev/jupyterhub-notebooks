import requests
import typing as typ
import pandas as pd
import json
from dataclasses import dataclass


@dataclass
class Info:
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

