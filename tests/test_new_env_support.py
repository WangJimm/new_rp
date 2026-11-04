"""Regression cases for environment selection; no SUMO process is started."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from rp.experiment.config import build_new_env_config, build_selected_env_config, validate_environment_inputs
from rp.experiment.scenario_catalog import load_scenario_catalog
from rp.experiment.scenario_loader import build_training_scenario, build_testing_scenario
from rp.experiment.scenario_selection import (build_training_scenario_schedule, iter_training_scenario_keys,
                                              iter_validation_scenario_keys, iter_testing_scenario_keys)
from rp.sumo.runtime import build_load_args

class NewEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / 'new_env'
        self.env.mkdir()
        self.rows = []
        for split, seed in (('training', 101), ('validation', 509), ('testing', 9001)):
            data = self.env / f'{split}_data'; data.mkdir()
            goals = self.env / f'{split}_data_goal'; goals.mkdir()
            general = data / f'output_2650_{seed}.rou.xml'
            general.write_text('<routes/>', encoding='utf-8')
            for p in ('p01', 'p02'):
                ev = data / f'ev_2650_{p}_{seed}.rou.xml'
                ev.write_text('<routes><vehicle id="EV_7"/><vehicle id="EV_2"/></routes>', encoding='utf-8')
                (goals / f'goal_ev_2650_{p}_{seed}.json').write_text(json.dumps(['1419061109','446290091']), encoding='utf-8')
                self.rows.append(dict(split=split, flow_veh_h='2650', background_seed=seed, penetration=p,
                    general_route_file=general.relative_to(self.env).as_posix(),
                    ev_route_file=ev.relative_to(self.env).as_posix(), ev_vehicle_count=2))
        self.write_manifest()
        (self.env/'env_static_tls.net.xml').write_text('<net><junction id="1419061109"/><junction id="446290091"/><junction id="5984003368"/></net>', encoding='utf-8')
        (self.env/'osm.sumocfg').write_text('<configuration><input><net-file value="env_static_tls.net.xml"/></input></configuration>', encoding='utf-8')

    def write_manifest(self):
        with (self.env/'scenario_manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(self.rows[0])); writer.writeheader(); writer.writerows(self.rows)

    def test_selector_defaults_to_original_and_explicit_new_env_uses_its_root(self):
        self.assertEqual(build_selected_env_config(self.root).environment.name, 'env')
        self.assertEqual(build_selected_env_config(self.root, 'new_env').paths.run_dir, self.env)
        with self.assertRaises(ValueError): build_selected_env_config(self.root, 'typo')

    def test_sparse_seed_partitions_are_exhaustive_disjoint_and_reproducible(self):
        config=build_new_env_config(self.root)
        for iterator,seed in ((iter_training_scenario_keys,101),(iter_validation_scenario_keys,509),(iter_testing_scenario_keys,9001)):
            keys=iterator(config)
            self.assertEqual(len(keys),2)
            self.assertEqual({key.seed for key in keys},{seed})
            self.assertEqual({key.probability_index for key in keys},{1,2})
        a=build_training_scenario_schedule(config,100,seed=42)
        self.assertEqual(a,build_training_scenario_schedule(config,100,seed=42))
        self.assertEqual({key.seed for key in a},{101})

    def test_real_xml_order_is_paired_with_goals_and_seed_is_sent_to_sumo(self):
        config=build_new_env_config(self.root)
        spec=build_training_scenario(config,0,1,101)
        self.assertEqual(spec.ev_ids,['EV_7','EV_2'])
        self.assertEqual(spec.goal_nodes,{'EV_7':'1419061109','EV_2':'446290091'})
        self.assertEqual(build_load_args(config,spec)[-2:],['--seed','101'])
        self.assertEqual(build_testing_scenario(config,0,2,9001).goal_file.parent,self.env/'testing_data_goal')
        with self.assertRaisesRegex(ValueError,'absent from training'):
            build_training_scenario(config,0,1,509)

    def test_preflight_finds_missing_goal_before_any_simulation(self):
        config=build_new_env_config(self.root)
        validate_environment_inputs(config)
        (self.env/'testing_data_goal'/'goal_ev_2650_p02_9001.json').unlink()
        with self.assertRaises(FileNotFoundError): validate_environment_inputs(config)

    def test_manifest_rejects_duplicate_cases_and_paths_outside_partition(self):
        self.rows.append(self.rows[0].copy()); self.write_manifest()
        with self.assertRaisesRegex(ValueError,'Duplicate'): load_scenario_catalog(self.env/'scenario_manifest.csv')
        self.rows.pop(); self.rows[0]['ev_route_file']='../outside.rou.xml'; self.write_manifest()
        with self.assertRaisesRegex(ValueError,'does not match'): load_scenario_catalog(self.env/'scenario_manifest.csv')

    def test_manifest_rejects_shared_seed_across_partitions(self):
        for row in self.rows:
            if row['split']=='validation':
                row.update(flow_veh_h='2800', background_seed=101,
                    general_route_file='validation_data/output_2800_101.rou.xml',
                    ev_route_file=f"validation_data/ev_2800_{row['penetration']}_101.rou.xml")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'more than one partition'): load_scenario_catalog(self.env/'scenario_manifest.csv')

if __name__=='__main__': unittest.main()
