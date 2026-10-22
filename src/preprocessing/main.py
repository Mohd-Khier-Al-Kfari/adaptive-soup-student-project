"""Student HAR workflow with dependency checks and bounded dataset jobs.

Author: Moh'd Khier Al Kfari.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import sys
sys.dont_write_bytecode = True
import time
import tomllib
from download import ROOT, inside, acquire, extract_existing, verify_archive
from preprocessing import (preprocess, load_parsed, inventory, parse_pamap2,
                           parse_realworld, parse_mhealth, channel_names)
from postprocessing import postprocess, manifest_for, prepare_payload, SPLITS

STAGES = ("download", "preprocess", "postprocess")
DATASETS = ("PAMAP2", "RealWorld", "MHEALTH")

def ensure_layout():
    """Create the portable project layout relative to this script."""
    base = ROOT / "data"
    for experiment in ("cross_dataset", "cross_gender", "cross_age"):
        (base / "prepared" / experiment / "metadata").mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        (base / "original_datasets" / dataset).mkdir(parents=True, exist_ok=True)
        (base / "processed_datasets").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)

def ask(message):
    """Only an explicit y/yes/n/no is accepted. EOF cancels; no default answer."""
    while True:
        answer = input(message + " [y/yes/n/no] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Invalid response. Enter exactly y, yes, n, or no.", flush=True)

def load_config(path):
    with inside(path).open("rb") as handle:
        config = tomllib.load(handle)
    p = config["processing"]
    for name in ("sample_rate_hz", "window_size", "shift"):
        if p[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if p["normalization"] != "training_domain":
        raise ValueError("Use training_domain; pass --source-statistics for frozen source moments.")
    if p["resampling"] != "linear":
        raise ValueError("Only the documented linear resampling implementation is available.")
    if p["tie_policy"] not in ("lowest_class_index", "reject"):
        raise ValueError("Unknown tie policy")
    for name in ("max_interpolation_gap_seconds", "max_timestamp_gap_seconds"):
        if p[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    for dataset in DATASETS:
        settings = config["datasets"][dataset]
        if not settings["url"].startswith(("http://", "https://")):
            raise ValueError(f"{dataset}: HTTP(S) download URL required")
        if not settings["archive"] or settings["archive"] != __import__("pathlib").Path(settings["archive"]).name:
            raise ValueError(f"{dataset}: archive must be a plain filename")
    return config

def configure_logging(write_file=True):
    logger = logging.getLogger("student_har")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if write_file:
        folder = inside(ROOT / "logs")
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(folder / "pipeline.log", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def event(logger, dataset, stage, outcome, message):
    logger.info("dataset=%s stage=%s outcome=%s %s", dataset, stage, outcome, message)

def extracted_files(dataset):
    root = inside(ROOT / "data" / "original_datasets" / dataset / "extracted")
    if not root.is_dir():
        return []
    if dataset == "PAMAP2":
        return [p for p in root.rglob("subject*.dat") if p.parent.name.lower() == "protocol"]
    if dataset == "MHEALTH":
        return list(root.rglob("*subject*.log"))
    return [p for p in root.rglob("*.zip")
            if "proband" in str(p).lower() and p.name.lower().startswith(("acc_", "gyr_", "mag_"))]

def parsed_path(dataset):
    return inside(ROOT / "data" / "processed_datasets" / f"{dataset}.pkl")

def write_report(dataset, kind, report):
    """Version reports; never replace previous inventory/coverage reports."""
    folder = inside(ROOT / "data" / "reports")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dataset}_{kind}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, indent=2,
                               default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)),
                    encoding="utf-8")
    return path

def coverage_report(payload, config, protocol, group, manifest):
    import numpy as np
    from postprocessing import ALIASES
    classes = config["protocols"][protocol].get("classes")
    invalid = {label.lower() for label in config["processing"]["invalid_labels"]}
    expected = set(classes) if classes else set()
    report = {"subjects": {}, "errors": [], "warnings": []}
    selected = ({manifest["temporal_subject"]} if "temporal_subject" in manifest else
                {s for name in SPLITS for s in manifest[name]})
    train_classes = set()
    for subject in sorted(selected):
        native = {}
        kept = {}
        potential_windows = 0
        for record in payload["subjects"].get(subject, []):
            labels = np.asarray(record["y"], dtype=str)
            for label, count in zip(*np.unique(labels, return_counts=True)):
                native[str(label)] = native.get(str(label), 0) + int(count)
                mapped = ALIASES.get(str(label), str(label)) if classes else str(label)
                if mapped.lower() not in invalid and (not classes or mapped in expected):
                    kept[mapped] = kept.get(mapped, 0) + int(count)
            from postprocessing import mapped as map_record
            for part in map_record(record, classes, invalid):
                n = len(part["x"])
                potential_windows += max(0, (n-config["processing"]["window_size"]) // config["processing"]["shift"]+1)
        report["subjects"][subject] = {"native_samples": native, "selected_samples": kept,
            "potential_windows_before_temporal_partition": potential_windows,
            "eligible_selected_activity": bool(kept)}
        if not kept or not potential_windows:
            report["errors"].append(f"{subject}: no usable selected-activity windows; manifest revision required")
        if classes and expected-set(kept):
            report["warnings"].append(f"{subject}: absent selected classes {sorted(expected-set(kept))}")
        if subject in manifest.get("train", []):
            train_classes.update(kept)
    if classes and "temporal_subject" not in manifest and expected-train_classes:
        report["errors"].append(f"Training coverage missing classes: {sorted(expected-train_classes)}")
    return report

def check_dataset(dataset, config, args, logger, save=True, deep=True):
    report = {"dataset": dataset, "protocol": args.protocol, "group": args.group,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "errors": [], "warnings": [], "archives": [],
              "known_coverage_findings": config.get("coverage_findings", {}).get(args.protocol, {}).get(dataset, {})}
    archive = inside(ROOT / "data" / "original_datasets" / dataset / config["datasets"][dataset]["archive"])
    if archive.exists():
        try:
            digest = verify_archive(archive, config["datasets"][dataset].get("sha256", ""))
            report["archives"].append({"path": str(archive), "sha256": digest, "status": "valid"})
        except Exception as error:
            report["errors"].append(f"Invalid archive: {error}")
    else:
        report["warnings"].append(f"Configured archive absent: {archive}")
    files = extracted_files(dataset)
    report["extracted_file_count"] = len(files)
    if not files:
        report["errors"].append(f"Dependency error: {dataset} extracted data absent/incomplete; download/extract before preprocess.")
    manifest = None
    try:
        # Check candidate structure without treating provisional as approved for execution.
        manifest = manifest_for(config, args.protocol, dataset, args.group, True)
        report["manifest"] = manifest
    except (ValueError, KeyError) as error:
        report["errors"].append(str(error))
    payload = None
    try:
        if parsed_path(dataset).exists():
            payload = load_parsed(dataset)
            report["parsed_inventory"] = inventory(payload)
            if payload["dataset"] != dataset or payload["channel_names"] != channel_names(dataset):
                raise ValueError("Parsed dataset identity/channel schema mismatch")
            if payload["processing"] != config["processing"]:
                raise ValueError("Parsed settings differ from current config; re-preprocess required")
        elif deep and files:
            parser = {"PAMAP2": parse_pamap2, "RealWorld": parse_realworld, "MHEALTH": parse_mhealth}[dataset]
            subjects, _ = parser(ROOT/"data"/"original_datasets"/dataset/"extracted",
                                config["datasets"][dataset], config["processing"])
            payload = {"dataset": dataset, "subjects": subjects, "channel_names": channel_names(dataset),
                       "processing": config["processing"], "source_files": {}}
            report["raw_inventory"] = inventory(payload)
            report["warnings"].append("Read-only raw parser audit; no parsed artifact was saved.")
        else:
            report["warnings"].append("Parsed data unavailable; coverage cannot be verified.")
        # A rejected known finding must not hide actual observed subject coverage.
        if manifest is None and payload is not None:
            section = config["protocols"][args.protocol].get(dataset, {})
            candidate = section if args.protocol == "cross_dataset" else section.get(args.group, {})
            if candidate and all(candidate.get(name) for name in SPLITS):
                manifest = candidate
        if payload is not None and manifest:
            report["coverage"] = coverage_report(payload, config, args.protocol, args.group, manifest)
            report["errors"].extend(report["coverage"]["errors"])
            report["warnings"].extend(report["coverage"]["warnings"])
            # Full window/temporal checks run only when the candidate passes known findings.
            if not report["errors"]:
                prepared = prepare_payload(payload, config, args.protocol, args.group, True)
                report["window_coverage"] = prepared["report"]
    except Exception as error:
        report["errors"].append(f"Data validation failed: {error}")
    report["ok"] = not report["errors"]
    if save:
        path = write_report(dataset, "check", report)
        event(logger, dataset, "check", "success" if report["ok"] else "blocked", f"report={path}")
    return report

def collect_choices(args, datasets):
    choices = {}
    for dataset in datasets:
        choices[dataset] = {}
        for stage in STAGES:
            if args.command not in ("all", stage):
                choices[dataset][stage] = False
            elif args.noninteractive:
                choices[dataset][stage] = (args.download if stage == "download" else
                                          args.command == stage or getattr(args, stage))
            else:
                choices[dataset][stage] = (stage == "download" and args.download) or ask(f"{dataset}: {stage}?")
    return choices

def plan_actions(args, datasets, choices=None):
    proposed = choices or {d: {s: bool(getattr(args, s)) for s in STAGES} for d in datasets}
    if choices is None and not any(any(x.values()) for x in proposed.values()):
        proposed = {d: dict.fromkeys(STAGES, True) for d in datasets}
    return [{"dataset": d, "actions": proposed[d],
             "preprocess_dependency": "available" if extracted_files(d) else
                 ("will require successful selected download/extraction" if proposed[d]["download"] else "BLOCKED: extracted data missing"),
             "postprocess_dependency": "available" if parsed_path(d).exists() else
                 ("will require successful selected preprocessing" if proposed[d]["preprocess"] else "BLOCKED: parsed data missing")}
            for d in datasets]

def postprocess_targets(dataset, args):
    """Return all experiments for the default run, or one explicitly requested target."""
    if args.protocol != "cross_dataset" or args.group is not None:
        return [(args.protocol, args.group)]
    targets=[("cross_dataset", None)]
    if dataset in ("PAMAP2", "RealWorld"):
        targets.extend(("cross_gender", group) for group in ("male", "female"))
    if dataset == "RealWorld":
        targets.extend(("cross_age", group) for group in ("below30", "age30plus"))
    return targets

def run_dataset(dataset, choices, config, args, logger):
    outcomes = {}
    for stage in STAGES:
        if not choices[stage]:
            outcomes[stage] = "not_selected"
            event(logger, dataset, stage, "not_selected", "No action")
            continue
        try:
            if stage == "preprocess" and not extracted_files(dataset):
                archive = ROOT / "data" / "original_datasets" / dataset / config["datasets"][dataset]["archive"]
                if archive.is_file():
                    event(logger, dataset, "download", "started", "Extracting existing local archive")
                    extract_existing(dataset, config["datasets"][dataset])
                    event(logger, dataset, "download", "success", "Existing local archive extracted")
            if stage == "preprocess" and not extracted_files(dataset):
                raise ValueError(f"Dependency error: {dataset} extracted data missing/incomplete; preprocess not started")
            if stage == "postprocess":
                if choices["preprocess"] and outcomes.get("preprocess") != "success":
                    raise ValueError("Dependency error: selected preprocessing failed; postprocess not started")
                if not parsed_path(dataset).exists():
                    raise ValueError(f"Dependency error: {dataset} parsed data missing; postprocess not started")
                targets=postprocess_targets(dataset,args)
                for protocol, group in targets:
                    manifest_for(config, protocol, dataset, group, args.accept_provisional)
                if args.source_statistics and not inside(args.source_statistics).is_file():
                    raise ValueError("Dependency error: source-statistics artifact must already exist")
            if stage == "preprocess" and choices["download"] and outcomes.get("download") != "success":
                raise ValueError("Dependency error: selected download/extraction failed; preprocess not started")
            event(logger, dataset, stage, "started", "Starting")
            if stage == "download":
                acquire(dataset, config["datasets"][dataset], allowed=True)
            elif stage == "preprocess":
                summary = preprocess(dataset, config)
                path = write_report(dataset, "inventory", summary)
                event(logger, dataset, stage, "inventory_saved", f"report={path}")
            else:
                def build_target(target):
                    protocol, group = target
                    label = f"{protocol}/{group or 'all'}"
                    event(logger, dataset, "postprocess", "started", label)
                    summary = postprocess(dataset, config, protocol, group,
                                          args.accept_provisional, args.source_statistics)
                    write_report(dataset, f"postprocess_{protocol}_{group or 'all'}", summary)
                    return label

                # Targets write to separate prepared folders and can safely run
                # concurrently after the parsed dataset is available.
                workers = min(max(1, args.workers), max(1, len(targets)))
                with ThreadPoolExecutor(max_workers=workers) as target_pool:
                    target_jobs = [target_pool.submit(build_target, target) for target in targets]
                    for target_job in target_jobs:
                        target_job.result()
            outcomes[stage] = "success"
            event(logger, dataset, stage, "success", "Completed")
        except Exception as error:
            outcomes[stage] = "failed"
            event(logger, dataset, stage, "failed", str(error))
            print(f"ERROR dataset={dataset} stage={stage}: {error}", file=sys.stderr, flush=True)
    return outcomes

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",nargs="?",default="all",
        choices=["plan","check","test","download","preprocess","postprocess","all","inventory","validate"])
    parser.add_argument("--dataset",action="append",choices=list(DATASETS))
    parser.add_argument("--config",default=str(ROOT/"src"/"preprocessing"/"config.toml"))
    parser.add_argument("--protocol",default="cross_dataset",choices=["cross_dataset","cross_gender","cross_age"])
    parser.add_argument("--group",choices=["male","female","below30","age30plus"])
    parser.add_argument("--workers",type=int,default=2,help="Bounded independent dataset jobs (1-3)")
    parser.add_argument("--noninteractive",action="store_true")
    for stage in STAGES:
        parser.add_argument("--"+stage,action="store_true")
    parser.add_argument("--accept-provisional",action="store_true",default=True,
                        help="Kept for compatibility; provisional manifests are accepted by default.")
    parser.add_argument("--source-statistics")
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config)
        if args.command != "plan":
            ensure_layout()
        if not 1<=args.workers<=3:
            raise ValueError("--workers must be between 1 and 3")
        datasets=list(dict.fromkeys(args.dataset or DATASETS))
        if args.command=="plan":
            # Absolutely no logging directory, report or other write in plan mode.
            report={"actions":plan_actions(args,datasets),"manifest_readiness":{}}
            for dataset in datasets:
                try:
                    report["manifest_readiness"][dataset]=manifest_for(config,args.protocol,dataset,args.group,True)
                except (ValueError,KeyError) as error:
                    report["manifest_readiness"][dataset]={"blocked":str(error)}
            print(json.dumps(report,indent=2))
            return 0
        logger=configure_logging()
        if args.command=="test":
            event(logger,"ALL","test","started","Synthetic data only")
            from tests.test_pipeline import synthetic_tests
            passed=synthetic_tests()
            event(logger,"ALL","test","success" if passed else "failed","Synthetic suite completed")
            return 0 if passed else 1
        if args.command in ("check","inventory"):
            failed=False
            for dataset in datasets:
                report=check_dataset(dataset,config,args,logger,deep=args.command=="check")
                print(json.dumps(report,indent=2))
                failed=failed or not report["ok"]
            return int(failed)
        if args.command=="validate":
            failed=False
            for dataset in datasets:
                try:
                    manifest=manifest_for(config,args.protocol,dataset,args.group,args.accept_provisional)
                    event(logger,dataset,"validate","success",json.dumps(manifest))
                except (ValueError,KeyError) as error:
                    failed=True
                    event(logger,dataset,"validate","blocked",str(error))
            return int(failed)
        if args.noninteractive and args.command=="all" and not any(getattr(args,s) for s in STAGES):
            raise ValueError("Noninteractive all needs explicit stage flags")
        if args.noninteractive and args.command=="download" and not args.download:
            raise ValueError("Noninteractive download requires --download")
        # All prompts finish before any selected stage starts.
        choices=collect_choices(args,datasets)
        print(json.dumps({"selected_plan":plan_actions(args,datasets,choices)},indent=2))
        active=[dataset for dataset in datasets if any(choices[dataset].values())]
        failed=False
        with ThreadPoolExecutor(max_workers=min(args.workers,max(1,len(active)))) as pool:
            jobs={pool.submit(run_dataset,d,choices[d],config,args,logger):d for d in active}
            for future in as_completed(jobs):
                result=future.result()
                failed=failed or "failed" in result.values()
        return int(failed)
    except (EOFError,KeyboardInterrupt):
        print("Cancelled before completing input; no selected jobs started.",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}",file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())

