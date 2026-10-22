"""Explicit manifests/windowing adapted from HAR data_pre_processing.py:110-291.
Copyright (C) 2024 Moh'd Khier Al Kfari; adaptations 2026.
SPDX-License-Identifier: GPL-3.0-or-later
No random-window splitting or target-label tuning.
Author: Moh'd Khier Al Kfari.
"""
import copy
import hashlib
import json
import pickle
from collections import Counter
import numpy as np
from download import ROOT, inside
from preprocessing import load_parsed, recording_segments

SPLITS = ("train","val","test")
ALIASES = {"Sitting and relaxing":"sitting","Standing still":"standing",
           "Lying down":"lying","Walking":"walking","Running":"running","Jogging":"running"}

def manifest_for(config, protocol, dataset, group=None, accept_provisional=False):
    section = config["protocols"][protocol]
    if dataset not in section:
        raise ValueError(f"{dataset} is not part of {protocol}")
    manifest = section[dataset]
    if protocol != "cross_dataset":
        if group not in manifest:
            raise ValueError(f"Choose a configured group: {list(manifest)}")
        manifest = manifest[group]
    if manifest.get("status",section.get("status")) == "unassigned":
        raise ValueError(f"{protocol}/{dataset}/{group}: split IDs remain UNASSIGNED; edit reviewed config first.")
    findings = config.get("coverage_findings", {}).get(protocol, {}).get(dataset, {})
    selected = {manifest["temporal_subject"]} if "temporal_subject" in manifest else {
        subject for name in SPLITS for subject in manifest.get(name, [])}
    blocked = [f"{subject}: {finding['reason']}" for subject, finding in findings.items()
               if subject in selected and finding.get("eligible") is False]
    if blocked:
        raise ValueError("Manifest revision required: " + "; ".join(blocked))
    if section.get("status") == "provisional" and not accept_provisional:
        raise ValueError("Candidate manifest is provisional. Audit coverage, then explicitly --accept-provisional.")
    if "temporal_subject" not in manifest:
        groups = [manifest.get(name,[]) for name in SPLITS]
        if not all(groups):
            raise ValueError("Each train/val/test subject list must be nonempty.")
        flat = sum(groups,[])
        if len(flat) != len(set(flat)):
            raise ValueError("Subject overlap/duplicate in explicit manifest")
        if "counts" in manifest and [len(x) for x in groups] != manifest["counts"]:
            raise ValueError("Configured subject counts do not match manifest")
    return manifest

def mapped(record, classes, invalid):
    record = copy.copy(record)
    native = np.asarray(record["y"],dtype=str)
    labels = np.asarray([ALIASES.get(y,y) if classes else y for y in native])
    keep = np.array([y.strip().lower() not in invalid and (not classes or y in classes) for y in labels])
    # Preserve gaps as separate records, never boolean-index and concatenate.
    from preprocessing import runs
    result = []
    for a,b in runs(keep):
        sub = dict(record,x=record["x"][a:b],y=labels[a:b],
                   time_seconds=record["time_seconds"][a:b],
                   segment_id=record["segment_id"]+f"/mapped{a}:{b}")
        result.append(sub)
    return result

def temporal_partitions(records, manifest, processing):
    """Per-activity raw intervals; re-clean each partition independently."""
    ratios = np.asarray(manifest["ratios"],float)
    if len(ratios)!=3 or (ratios<=0).any() or not np.isclose(ratios.sum(),1):
        raise ValueError("Temporal ratios must be three positive values summing to 1")
    by_class = {}
    for record in records:
        if "raw_x" not in record:
            raise ValueError("Temporal exception needs raw-support payload; rerun parser with current code.")
        labels = set(record["y"])
        if len(labels)!=1:
            raise ValueError("Temporal allocation requires contiguous single-activity segments")
        by_class.setdefault(str(record["y"][0]),[]).append(record)
    output = {name:[] for name in SPLITS}
    for label, activity_records in sorted(by_class.items()):
        activity_records = sorted(activity_records,key=lambda r:(r["recording_id"],r["raw_time_seconds"][0]))
        total = sum(len(r["raw_x"]) for r in activity_records)
        bounds = [0,int(total*ratios[0]),int(total*(ratios[0]+ratios[1])),total]
        if min(np.diff(bounds))<=0:
            raise ValueError(f"{label}: not enough raw samples for temporal partitions")
        offset=0
        for record in activity_records:
            for name,left,right in zip(SPLITS,bounds[:-1],bounds[1:]):
                a,b=max(0,left-offset),min(len(record["raw_x"]),right-offset)
                if a<b:
                    parts = recording_segments(record["subject_id"],record["recording_id"],
                        record["raw_time_seconds"][a:b],record["raw_x"][a:b],
                        np.repeat(label,b-a),record["channel_names"],processing)
                    for part in parts:
                        part["segment_id"] += "/temporal_"+name
                        part["partition"] = name
                    output[name].extend(parts)
            offset += len(record["raw_x"])
    return output

def windows(records, label_map, size, shift, tie_policy):
    if size<=0 or shift<=0:
        raise ValueError("Window size/shift must be positive")
    if tie_policy not in ("lowest_class_index","reject"):
        raise ValueError("tie_policy must be lowest_class_index or reject")
    xs,ys,ids,starts,ends,segments,purity = [],[],[],[],[],[],[]
    for record in records:
        for start in range(0,len(record["x"])-size+1,shift):
            stop=start+size
            labels=np.array([label_map[y] for y in record["y"][start:stop]],dtype=np.int64)
            counts=np.bincount(labels,minlength=len(label_map))
            winners=np.flatnonzero(counts==counts.max())
            if len(winners)>1 and tie_policy=="reject":
                continue
            xs.append(record["x"][start:stop])
            ys.append(int(winners[0]))
            ids.append(record["recording_id"])
            segments.append(record["segment_id"])
            starts.append(float(record["time_seconds"][start]))
            ends.append(float(record["time_seconds"][stop-1]))
            purity.append(float(counts.max()/size))
    channels=records[0]["x"].shape[1] if records else 0
    return {"x":np.stack(xs).astype(np.float32) if xs else np.empty((0,size,channels),np.float32),
        "y":np.array(ys,dtype=np.int64),"recording_id":np.array(ids,dtype=str),
        "segment_id":np.array(segments,dtype=str),"start_time_seconds":np.array(starts),
        "end_time_seconds":np.array(ends),"label_purity":np.array(purity,dtype=np.float32)}

def prepare_payload(payload, config, protocol="cross_dataset", group=None,
                    accept_provisional=False, source_statistics=None):
    dataset=payload["dataset"]
    manifest=manifest_for(config,protocol,dataset,group,accept_provisional)
    settings=config["processing"]
    if payload["processing"] != settings:
        raise ValueError("Parsed processing settings differ; rerun preprocessing before postprocessing.")
    classes=config["protocols"][protocol].get("classes")
    invalid={s.strip().lower() for s in settings["invalid_labels"]}
    filtered={subject:[part for record in records for part in mapped(record,classes,invalid)]
              for subject,records in payload["subjects"].items()}
    if "temporal_subject" in manifest:
        subject=manifest["temporal_subject"]
        if protocol!="cross_gender" or dataset!="PAMAP2" or group!="female" or subject!="subject102":
            raise ValueError("Temporal exception permitted only for PAMAP2/female cross_gender")
        if subject not in filtered or not filtered[subject]:
            raise ValueError("Female102 has no eligible recordings")
        # Raw labels must receive the same mapping/filtering as retained records.
        split_records={name:{subject:rs} for name,rs in temporal_partitions(filtered[subject],manifest,settings).items()}
    else:
        missing=[s for name in SPLITS for s in manifest[name] if s not in filtered or not filtered[s]]
        if missing:
            raise ValueError(f"Configured subjects lack usable selected activities: {missing}")
        split_records={name:{s:filtered[s] for s in manifest[name]} for name in SPLITS}
    label_names=list(classes) if classes else sorted({str(y) for records in filtered.values() for r in records for y in r["y"]})
    label_map={label:i for i,label in enumerate(label_names)}
    channels=payload["channel_names"]
    for subjects in split_records.values():
        for records in subjects.values():
            for r in records:
                if r["channel_names"]!=channels or not np.isfinite(r["x"]).all():
                    raise ValueError("Nonfinite data or inconsistent channel schema")
    raw_train=[r["x"] for records in split_records["train"].values() for r in records]
    if not raw_train:
        raise ValueError("No training records")
    data=np.concatenate(raw_train).astype(np.float64)
    if source_statistics is None:
        mean,std=data.mean(axis=0),data.std(axis=0)
        provenance={"fit_subjects":list(split_records["train"]),"fit_partition":"train",
                    "fit_dataset":dataset,"fit_group":group,"fit_protocol":protocol,
                    "fit_sample_count":len(data),"mode":"training_domain"}
    else:
        if source_statistics["channel_names"]!=channels:
            # Exact schema equality is required; no implicit hand/forearm/shin relabeling.
            raise ValueError("Source-statistics channel names differ; approve an explicit semantic mapping first")
        mean=np.asarray(source_statistics["mean"])
        std=np.asarray(source_statistics["std"])
        provenance={"mode":"source_training","source_provenance":source_statistics["provenance"]}
    if mean.shape!=(len(channels),) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<0):
        raise ValueError("Invalid normalization moments")
    constant=np.flatnonzero(std==0).tolist()
    std=np.where(std==0,1,std)
    # Normalize continuous recordings before creating windows.
    normalized_records={}
    for name, subjects in split_records.items():
        normalized_records[name]={}
        for subject, records in subjects.items():
            normalized_records[name][subject]=[
                dict(record, x=((record["x"].astype(np.float64)-mean)/std).astype(np.float32))
                for record in records
            ]
    output={}
    report={}
    for name,subjects in split_records.items():
        output[name]={}
        report[name]={}
        for subject,records in normalized_records[name].items():
            bundle=windows(records,label_map,int(settings["window_size"]),int(settings["shift"]),settings["tie_policy"])
            if not len(bundle["y"]):
                raise ValueError(f"{name}/{subject}: no full windows; review coverage before freezing manifest")
            if not np.isfinite(bundle["x"]).all():
                raise ValueError(f"{name}/{subject}: normalization produced nonfinite features")
            bundle["subject_id"]=np.repeat(subject,len(bundle["y"]))
            output[name][subject]=bundle
            report[name][subject]={"windows":len(bundle["y"]),
                "class_windows":dict(Counter(label_names[i] for i in bundle["y"]))}
    # Shuffle complete windows deterministically while preserving paired fields.
    rng=np.random.default_rng(int(settings.get("shuffle_seed", 52)))
    for subjects in output.values():
        for bundle in subjects.values():
            order=rng.permutation(len(bundle["y"]))
            for key, value in bundle.items():
                if hasattr(value, "__len__") and len(value)==len(order):
                    bundle[key]=value[order]
    train_labels=np.concatenate([b["y"] for b in output["train"].values()])
    missing=set(label_map.values())-set(train_labels.tolist())
    if missing:
        raise ValueError(f"Training windows lack classes: {[label_names[i] for i in missing]}")
    # Verify interval disjointness even for the female exception.
    intervals={}
    for name,subjects in split_records.items():
        for subject,records in subjects.items():
            for record in records:
                key=(subject,record["recording_id"])
                begin,end=record["time_seconds"][[0,-1]]
                for other,a,b in intervals.get(key,[]):
                    if other!=name and max(a,begin)<=min(b,end):
                        raise ValueError("Raw/time interval overlap across partitions")
                intervals.setdefault(key,[]).append((name,begin,end))
    return {"schema_version":1,"dataset":dataset,"protocol":protocol,"group":group,
        "status":"candidate_manifest_used" if accept_provisional else "reviewed_manifest",
        "splits":output,"manifest":manifest,"label_map":label_map,"channel_names":channels,
        "normalization":{"mean":mean,"std":std,"constant_channel_indices":constant,
                         "channel_names":channels,"provenance":provenance},
        "source_files":payload.get("source_files",{}),"processing":settings,"report":report,
        "target_usage":"train features only for adaptation; all true labels eval sidecar; val reserved"}

def postprocess(dataset,config,protocol,group=None,accept_provisional=False,source_statistics=None):
    source=None
    if source_statistics:
        with inside(source_statistics).open("rb") as handle:
            source=pickle.load(handle)["normalization"]
    result=prepare_payload(load_parsed(dataset),config,protocol,group,accept_provisional,source)
    folder = ROOT / "data" / "prepared" / protocol / dataset
    if group is not None:
        folder = folder / group
    folder = inside(folder)
    folder.mkdir(parents=True,exist_ok=True)
    # Separate true labels, including target-training truth, from adaptation features.
    labels={name:{subject:bundle.pop("y") for subject,bundle in subjects.items()}
            for name,subjects in result["splits"].items()}
    from download import checksum
    code_hashes={name:checksum(ROOT/"src"/"preprocessing"/name)
                 for name in ("main.py","download.py","preprocessing.py","postprocessing.py")}
    result["code_sha256"]=code_hashes
    fingerprint=hashlib.sha256(json.dumps({"config":config,"sources":result["source_files"],"code":code_hashes,
                                          "normalization_provenance":result["normalization"]["provenance"],
                                          "normalization_mean":result["normalization"]["mean"].tolist(),
                                          "normalization_std":result["normalization"]["std"].tolist()},
                                          sort_keys=True).encode()).hexdigest()
    result["fingerprint"]=fingerprint
    # Student-facing features contain only sensor arrays grouped by split/subject.
    feature_data={name:{subject:bundle["x"] for subject,bundle in subjects.items()}
                  for name,subjects in result["splits"].items()}
    with (folder/"features.pkl").open("wb") as handle:
        pickle.dump(feature_data,handle,pickle.HIGHEST_PROTOCOL)
    with (folder/"evaluation_labels.pkl").open("wb") as handle:
        pickle.dump({"label_map":result["label_map"],"labels":labels,"fingerprint":fingerprint},handle,pickle.HIGHEST_PROTOCOL)
    # Keep student-facing metadata small and put it at the experiment root.
    # Detailed inventories remain under data/reports for diagnostics.
    metadata={
        "dataset": result["dataset"],
        "experiment": result["protocol"],
        "group": result["group"],
        "splits": result["manifest"],
        "activities": result["label_map"],
        "channels": result["channel_names"],
        "sampling_rate_hz": config["processing"]["sample_rate_hz"],
        "window_size": config["processing"]["window_size"],
        "shift": config["processing"]["shift"],
        "normalization": result["normalization"],
        "fingerprint": result["fingerprint"],
    }
    metadata_folder=inside(ROOT/"data"/"prepared"/protocol/"metadata")
    metadata_folder.mkdir(parents=True,exist_ok=True)
    metadata_name=dataset if not group else f"{dataset}_{group}"
    (metadata_folder/f"{metadata_name}.json").write_text(
        json.dumps(metadata,indent=2,default=lambda value: value.tolist() if hasattr(value,"tolist") else value),
        encoding="utf-8")
    # Remove the previous verbose per-dataset metadata artifact when present.
    (folder/"metadata.json").unlink(missing_ok=True)
    return metadata

