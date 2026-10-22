"""Synthetic pipeline tests. Author: Moh'd Khier Al Kfari."""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_REPO / "src" / "preprocessing"))
from main import *
def synthetic_tests():
    """Small in-memory checks; no downloads or real datasets."""
    import copy
    import unittest
    from unittest.mock import patch
    from io import BytesIO
    import zipfile
    import numpy as np
    import preprocessing as pre
    import postprocessing as post

    class Smoke(unittest.TestCase):
        def setUp(self):
            self.config = copy.deepcopy(load_config(ROOT/"src"/"preprocessing"/"config.toml"))
        def rec(self, subject, label="walking", value=2, n=600):
            return {"subject_id": subject, "recording_id": f"{subject}/{label}", "segment_id": f"{subject}/{label}/0",
                    "x": np.full((n,18),value,np.float32), "y": np.repeat(label,n),
                    "time_seconds": np.arange(n)/50, "channel_names": pre.channel_names("PAMAP2")}
        def test_strict_input(self):
            with patch("builtins.input",side_effect=["yy", "", "YES", "maybe", "No"]) as prompt:
                self.assertTrue(ask("download"))
                self.assertFalse(ask("download"))
                self.assertEqual(prompt.call_count,5)
        def test_known_subject109_block(self):
            manifest = post.manifest_for(self.config,"cross_dataset","PAMAP2",accept_provisional=True)
            selected = sum((manifest.get(split, []) for split in SPLITS), [])
            self.assertNotIn("subject109", selected)
        def test_subject109_observed_coverage(self):
            args = argparse.Namespace(protocol="cross_dataset",group=None)
            manifest = self.config["protocols"]["cross_dataset"]["PAMAP2"]
            payload={"subjects":{"subject109":[self.rec("subject109","rope jumping")]}}
            report=coverage_report(payload,self.config,args.protocol,None,manifest)
            self.assertFalse(report["subjects"]["subject109"]["eligible_selected_activity"])
            self.assertTrue(any("subject109" in e for e in report["errors"]))
        def test_training_moments_and_overlap(self):
            self.config.pop("coverage_findings",None)  # hypothetical complete synthetic cohort, not production override
            subjects={}
            for sid in range(101,110):
                s=f"subject{sid}"
                subjects[s]=[self.rec(s,label,100 if sid in (102,103,104,105) else 2)
                             for label in self.config["protocols"]["cross_dataset"]["classes"]]
            payload={"dataset":"PAMAP2","subjects":subjects,"processing":self.config["processing"],
                     "channel_names":pre.channel_names("PAMAP2"),"source_files":{}}
            out=post.prepare_payload(payload,self.config,accept_provisional=True)
            self.assertTrue(np.isfinite(out["normalization"]["mean"]).all())
            self.assertTrue(np.isfinite(out["splits"]["val"]["subject105"]["x"]).all())
            self.config["protocols"]["cross_dataset"]["PAMAP2"]["val"]=["subject101"]
            with self.assertRaisesRegex(ValueError,"overlap"):
                post.prepare_payload(payload,self.config,accept_provisional=True)
        def test_boundaries_and_ties(self):
            short=self.rec("s",n=149)
            self.assertEqual(len(post.windows([short,short],{"walking":0},150,75,"reject")["y"]),0)
            r=self.rec("s",n=150)
            r["y"]=np.array(["walking"]*75+["lying"]*75)
            self.assertEqual(post.windows([r],{"lying":0,"walking":1},150,75,"lowest_class_index")["y"].tolist(),[0])
            self.assertEqual(len(post.windows([r],{"lying":0,"walking":1},150,75,"reject")["y"]),0)
        def test_temporal_exception(self):
            records=[]
            for label in ("lying","sitting"):
                t=np.arange(4000)/50
                records.extend(pre.recording_segments("subject102","subject102/"+label,t,
                    np.ones((4000,18)),np.repeat(label,4000),pre.channel_names("PAMAP2"),self.config["processing"]))
            manifest=self.config["protocols"]["cross_gender"]["PAMAP2"]["female"]
            parts=post.temporal_partitions(records,manifest,self.config["processing"])
            for label in ("lying","sitting"):
                intervals=[]
                for name in SPLITS:
                    rs=[r for r in parts[name] if r["y"][0]==label]
                    self.assertTrue(rs)
                    intervals.append((rs[0]["time_seconds"][0],rs[-1]["time_seconds"][-1]))
                self.assertTrue(intervals[0][1]<intervals[1][0]<intervals[1][1]<intervals[2][0])
        def test_missing_dependencies(self):
            config=self.config
            args=argparse.Namespace(protocol="cross_dataset",group=None,accept_provisional=True,source_statistics=None)
            logger=logging.getLogger("smoke")
            with patch("main.extracted_files",return_value=[]), patch("main.preprocess") as fn:
                result=run_dataset("RealWorld",{"download":False,"preprocess":True,"postprocess":False},config,args,logger)
                self.assertEqual(result["preprocess"],"failed")
                fn.assert_not_called()
        def test_check_realworld_dependency(self):
            args=argparse.Namespace(protocol="cross_dataset",group=None)
            with patch("main.extracted_files",return_value=[]), patch("main.parsed_path") as path:
                path.return_value.exists.return_value=False
                report=check_dataset("RealWorld",self.config,args,logging.getLogger("smoke"),save=False,deep=False)
            self.assertFalse(report["ok"])
            self.assertTrue(any("RealWorld extracted data" in e for e in report["errors"]))
        def test_collect_before_execute_and_parallel(self):
            args=argparse.Namespace(command="all",noninteractive=False,download=False)
            with patch("builtins.input",side_effect=["no"]*9):
                choices=collect_choices(args,list(DATASETS))
            self.assertEqual(len(choices),3)
            self.assertFalse(any(any(v.values()) for v in choices.values()))
            import threading
            barrier=threading.Barrier(2)
            def independent(d):
                barrier.wait(timeout=5)
                return d
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks=[pool.submit(independent,d) for d in DATASETS[:2]]
                self.assertEqual([f.result() for f in tasks],list(DATASETS[:2]))
        def test_actual_parallel_scheduler(self):
            import threading
            barrier=threading.Barrier(2)
            def fake_preprocess(dataset, config):
                barrier.wait(timeout=5)
                return {"dataset":dataset}
            with (patch("main.configure_logging",return_value=logging.getLogger("smoke")),
                  patch("main.extracted_files",return_value=[ROOT/"fixture"]),
                  patch("main.preprocess",side_effect=fake_preprocess) as fn,
                  patch("main.write_report",return_value="in_memory_report")):
                code=main(["all","--dataset","PAMAP2","--dataset","MHEALTH",
                           "--noninteractive","--preprocess","--workers","2"])
                self.assertEqual(code,0)
                self.assertEqual(fn.call_count,2)
        def test_zip_and_download_guards(self):
            with self.assertRaises(ValueError):
                acquire("PAMAP2",self.config["datasets"]["PAMAP2"])
            stream=BytesIO()
            with zipfile.ZipFile(stream,"w") as z:
                z.writestr("../escape.txt","bad")
            stream.seek(0)
            from pathlib import Path
            from download import extract_archive
            with patch.object(Path,"mkdir"):
                with self.assertRaisesRegex(ValueError,"Unsafe"):
                    extract_archive(stream,ROOT/"unused")
        def test_plan_is_read_only(self):
            args=argparse.Namespace(download=False,preprocess=False,postprocess=False)
            with patch("main.extracted_files",return_value=[]):
                result=plan_actions(args,["RealWorld"])
            self.assertIn("require successful",result[0]["preprocess_dependency"])
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Smoke))
    return result.wasSuccessful()


