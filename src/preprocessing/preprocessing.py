"""Standalone HAR parsers: adapted from SelfHAR raw_data_processing.py.
Provenance: keep_cols:64, PAMAP2:279, MHEALTH multi-sensor:392,
RealWorld:766; recording boundaries and bounded interpolation added.
Copyright (C) 2024 Moh'd Khier Al Kfari; adaptations 2026.
SPDX-License-Identifier: GPL-3.0-or-later
Author: Moh'd Khier Al Kfari.
"""
from pathlib import Path
from io import BytesIO
import json
import pickle
import re
import zipfile
import numpy as np
import pandas as pd
from download import ROOT, inside, checksum

PAM_LABELS = {0:"other (transient activities)",1:"lying",2:"sitting",3:"standing",
              4:"walking",5:"running",6:"cycling",7:"Nordic walking",
              12:"ascending stairs",13:"descending stairs",16:"vacuum cleaning",
              17:"ironing",24:"rope jumping",9:"watching TV",10:"computer work",
              11:"car driving",18:"folding laundry",19:"house cleaning",20:"playing soccer"}
MH_LABELS = dict(enumerate(["null","Standing still","Sitting and relaxing","Lying down",
    "Walking","Climbing stairs","Waist bends forward","Frontal elevation of arms",
    "Knees bending (crouching)","Cycling","Jogging","Running","Jump front & back"]))
RW_LABELS = ["climbingdown","climbingup","jumping","lying","running","sitting","standing","walking"]

def channel_names(dataset):
    positions = {"PAMAP2":["hand","ankle"], "RealWorld":["forearm","shin"],
                 "MHEALTH":["forearm","ankle"]}[dataset]
    return [f"{sensor}_{axis}_{position}" for position in positions
            for sensor in ("acc","gyr","mag") for axis in ("x","y","z")]

def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def recording_segments(subject, identifier, times, values, labels, channels, settings):
    """Split gaps/label boundaries BEFORE interpolation; never stitch removed runs."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=str)
    if x.shape != (len(t), len(channels)) or len(y) != len(t):
        raise ValueError(f"{identifier}: incompatible time/channel/label dimensions")
    if not len(t):
        return []
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{identifier}: timestamps must be finite and strictly increasing")
    breaks = np.r_[0, np.flatnonzero((np.diff(t) > settings["max_timestamp_gap_seconds"])
                                    | (y[1:] != y[:-1])) + 1, len(t)]
    result = []
    for section, (a,b) in enumerate(zip(breaks[:-1], breaks[1:])):
        tx, xx = t[a:b], x[a:b].copy()
        for col in range(xx.shape[1]):
            for start, stop in runs(~np.isfinite(xx[:,col])):
                if start > 0 and stop < len(xx) and tx[stop]-tx[start-1] <= settings["max_interpolation_gap_seconds"]:
                    xx[start:stop,col] = np.interp(tx[start:stop], tx[[start-1,stop]], xx[[start-1,stop],col])
        for block, (start, stop) in enumerate(runs(np.isfinite(xx).all(axis=1))):
            tt, vv = tx[start:stop], xx[start:stop]
            if not len(tt):
                continue
            grid = np.arange(int(np.floor((tt[-1]-tt[0])*settings["sample_rate_hz"] + 1e-7))+1) / settings["sample_rate_hz"] + tt[0]
            if settings["resampling"] != "linear":
                raise ValueError("Only explicit linear interpolation is implemented; no implicit filter change.")
            out = np.column_stack([np.interp(grid,tt,vv[:,c]) for c in range(vv.shape[1])])
            record = {"subject_id":subject, "recording_id":identifier,
                "segment_id":f"{identifier}/segment{section}/valid{block}",
                "x":out.astype(np.float32), "y":np.repeat(y[a],len(grid)),
                "time_seconds":grid, "channel_names":list(channels)}
            if subject == "subject102":
                record["raw_x"] = x[a:b][start:stop].copy()
                record["raw_time_seconds"] = tt.copy()
            result.append(record)
    return result

def parse_pamap2(root, dataset_settings, processing):
    result, files = {}, []
    paths = sorted(Path(root).rglob("subject*.dat"))
    for path in paths:
        if path.parent.name.lower() not in ({"protocol","optional"} if dataset_settings["include_optional"] else {"protocol"}):
            continue
        subject = path.stem.lower()
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        # SelfHAR selects 16g acceleration, gyro and mag; chest and acc6 excluded.
        indices = [4,5,6,10,11,12,13,14,15,38,39,40,44,45,46,47,48,49]
        if frame.shape[1] < 54:
            raise ValueError(f"{path}: expected PAMAP2's 54 columns")
        labels = [PAM_LABELS.get(int(v),"unknown") for v in frame[1]]
        rec_id = f"{subject}/{path.parent.name}/{path.name}"
        result.setdefault(subject,[]).extend(recording_segments(subject,rec_id,frame[0],
            frame[indices],labels,channel_names("PAMAP2"),processing))
        files.append(path)
    return result, files

def parse_mhealth(root, dataset_settings, processing):
    result, files = {}, []
    for path in sorted(Path(root).rglob("*.log")):
        match = re.search(r"subject(\d+)\.log$",path.name,re.I)
        if not match:
            continue
        subject = f"subject{int(match[1])}"
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] != 24:
            raise ValueError(f"{path}: expected 24 MHEALTH columns")
        # Raw right-lower-arm block (14:23), then left-ankle block (5:14).
        indices = list(range(14,23)) + list(range(5,14))
        labels = [MH_LABELS.get(int(v),"unknown") for v in frame[23]]
        times = np.arange(len(frame)) / dataset_settings["sampling_rate_hz"]
        if subject in result:
            raise ValueError(f"Duplicate MHEALTH file for {subject}; resolve provenance")
        result[subject] = recording_segments(subject,path.name,times,frame[indices],
            labels,channel_names("MHEALTH"),processing)
        files.append(path)
    return result, files

def zip_csvs(source, prefix="", depth=0):
    if depth > 4:
        raise ValueError("Unexpectedly deep RealWorld nested archive")
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            name = prefix + "/" + item.filename
            if item.filename.lower().endswith(".csv"):
                yield name, archive.read(item)
            elif item.filename.lower().endswith(".zip"):
                yield from zip_csvs(BytesIO(archive.read(item)),name,depth+1)

def parse_realworld(root, dataset_settings, processing):
    groups, files = {}, []
    for path in sorted(Path(root).rglob("*.zip")):
        sm = re.search(r"proband(\d+)",str(path),re.I)
        modality = re.match(r"(acc|gyr|mag)_",path.name,re.I)
        if not sm or not modality or "sqlite" in path.name.lower():
            continue
        files.append(path)
        subject, sensor = f"proband{int(sm[1])}", modality[1].lower()
        for name, content in zip_csvs(path):
            position = re.search(r"(forearm|shin)(?:[_.]|$)",name,re.I)
            activity = next((label for label in RW_LABELS if re.search(r"(?:^|[/_])"+label+r"(?:[_/.]|$)",name,re.I)),None)
            if not position or not activity:
                continue
            if subject == "proband2" and activity == "climbingup" and dataset_settings["preserve_proband2_climbingup_exclusion"]:
                continue
            # Keep repeated numbered trials distinct; merge modalities only within that trial.
            trial_match = re.search(activity+r"_(\d+)(?:[_/.]|$)",name,re.I)
            trial = trial_match[1] if trial_match else "1"
            key = (subject,activity,trial)
            stream = (sensor,position[1].lower())
            frame = pd.read_csv(BytesIO(content))
            required = ["attr_time","attr_x","attr_y","attr_z"]
            if not set(required).issubset(frame):
                raise ValueError(f"{path}:{name}: missing {required}")
            frame = frame[required].apply(pd.to_numeric,errors="coerce").sort_values("attr_time")
            if stream in groups.setdefault(key,{}):
                raise ValueError(f"Duplicate RealWorld stream {key}/{stream}; ambiguous trial identity")
            groups[key][stream] = frame
    result = {}
    for (subject,activity,trial), streams in sorted(groups.items()):
        ordered = [(sensor,pos) for pos in ("forearm","shin") for sensor in ("acc","gyr","mag")]
        if set(streams) != set(ordered):
            raise ValueError(f"{subject}/{activity}/{trial}: incomplete six-stream IMU schema")
        starts = [f.attr_time.min()/1000 for f in streams.values()]
        ends = [f.attr_time.max()/1000 for f in streams.values()]
        begin, end = max(starts), min(ends)
        if not np.isfinite([begin,end]).all() or end <= begin:
            raise ValueError(f"{subject}/{activity}: no shared sensor time range")
        grid = np.arange(int((end-begin)*processing["sample_rate_hz"])+1)/processing["sample_rate_hz"]+begin
        arrays = []
        for stream in ordered:
            frame = streams[stream]
            t = frame.attr_time.to_numpy()/1000
            if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                raise ValueError(f"{subject}/{activity}/{stream}: duplicate/invalid timestamps")
            axis_values = []
            for axis in ("x","y","z"):
                v = frame["attr_"+axis].to_numpy()
                valid = np.isfinite(v)
                if valid.sum() < 2:
                    raise ValueError(f"{subject}/{activity}/{stream}: insufficient valid samples")
                tt, vv = t[valid],v[valid]
                out = np.interp(grid,tt,vv,left=np.nan,right=np.nan)
                right = np.searchsorted(tt,grid,side="left")
                left = np.maximum(right-1,0)
                clipped = np.minimum(right,len(tt)-1)
                exact = np.isclose(grid,tt[clipped],rtol=0,atol=1e-7)
                out[(tt[clipped]-tt[left] > processing["max_interpolation_gap_seconds"]) & ~exact] = np.nan
                axis_values.append(out)
            arrays.append(np.column_stack(axis_values))
        rec_id = f"{subject}/{activity}/trial{trial}"
        # Do not fill alignment gaps again; preserve invalid runs.
        clean_settings = dict(processing,max_interpolation_gap_seconds=0)
        result.setdefault(subject,[]).extend(recording_segments(subject,rec_id,grid,
            np.column_stack(arrays),np.repeat(activity,len(grid)),channel_names("RealWorld"),clean_settings))
    return result, files

def preprocess(dataset, config):
    settings = config["datasets"][dataset]
    root = inside(ROOT / "data" / "original_datasets" / dataset / "extracted")
    if not root.exists():
        raise FileNotFoundError(f"Extracted data absent: {root}. Download explicitly or extract there.")
    parser = {"PAMAP2":parse_pamap2,"RealWorld":parse_realworld,"MHEALTH":parse_mhealth}[dataset]
    subjects, paths = parser(root, settings, config["processing"])
    if not subjects or not any(subjects.values()):
        raise ValueError(f"No valid {dataset} recordings found in {root}")
    payload = {"schema_version":1,"dataset":dataset,"subjects":subjects,
        "channel_names":channel_names(dataset),"processing":config["processing"],
        "source_files":{str(p.relative_to(ROOT)):checksum(p) for p in paths},
        "units_policy":"native units retained; no implicit calibration or gravity subtraction"}
    target = inside(ROOT / "data" / "processed_datasets" / f"{dataset}.pkl")
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(payload,handle,pickle.HIGHEST_PROTOCOL)
    summary = inventory(payload)
    target.with_suffix(".metadata.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

def load_parsed(dataset):
    path = inside(ROOT / "data" / "processed_datasets" / f"{dataset}.pkl")
    # Only load trusted local pipeline artifacts: pickle is executable serialization.
    with path.open("rb") as handle:
        return pickle.load(handle)

def inventory(payload):
    return {"dataset":payload["dataset"],"channel_names":payload["channel_names"],
        "subjects":{subject:{"segments":len(records),"samples":sum(len(r["x"]) for r in records),
            "class_samples":{label:sum(int(np.sum(r["y"]==label)) for r in records)
                             for label in sorted({str(y) for r in records for y in r["y"]})}}
                    for subject,records in payload["subjects"].items()}}

