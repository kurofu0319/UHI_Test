from __future__ import annotations

import unittest
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TimeSeriesWorkflowTest(unittest.TestCase):
    def test_all_workflow_branches_use_multivariate_inputs(self):
        workflow = (ROOT / "scripts" / "_baseline_workflows.bash").read_text()
        full_period = (
            ROOT / "scripts" / "prepare_full_period_figure_data.py"
        ).read_text()
        for source in (workflow, full_period):
            self.assertNotIn("--features S", source)
            self.assertNotIn('"--features", "S"', source)
            self.assertIn("features M", source.replace('"--features", "M"', "features M"))
            self.assertIn("24", source)
        self.assertIn("--evaluate_target_only", workflow)
        self.assertIn("--target_only_metrics", workflow)
        self.assertNotIn("global_scaler", workflow)
        self.assertIn("multivariate_24.schema", workflow)

    def test_stmtm_standardizes_pretraining_and_finetuning(self):
        filternet_loader = (
            ROOT / "baseline/filternet/data_provider/data_loader.py"
        ).read_text()
        stmtm_loader = (
            ROOT / "baseline/st_mtm/data_provider/data_loader.py"
        ).read_text()
        stmtm_model = (ROOT / "baseline/st_mtm/models/STMTM.py").read_text()
        stmtm_factory = (
            ROOT / "baseline/st_mtm/data_provider/data_factory.py"
        ).read_text()
        self.assertIn("target='OT', scale=False", filternet_loader)
        self.assertIn("target='OT', scale=True", stmtm_loader)
        self.assertEqual(stmtm_factory.count("scale=True"), 2)
        self.assertNotIn("use_mean_normalization", stmtm_model)

    def test_validation_window_produces_36_targets(self):
        for relative in (
            "baseline/filternet/data_provider/data_loader.py",
            "baseline/st_mtm/data_provider/data_loader.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn(
                "between('2017-01-01', '2020-12-01')", source
            )

    def test_stmtm_windows_cover_each_target_month_once(self):
        script = textwrap.dedent("""
            import tempfile
            from pathlib import Path

            import numpy as np
            import pandas as pd
            from data_provider.data_loader import Dataset_Custom

            with tempfile.TemporaryDirectory() as directory:
                dates = pd.date_range('2003-01-01', '2020-12-01', freq='MS')
                frame = pd.DataFrame({'date': dates})
                for i in range(23):
                    frame[f'feature_{i}'] = np.arange(len(dates)) + i
                frame['Amod2'] = np.arange(len(dates))
                frame.to_csv(Path(directory) / 'city.csv', index=False)

                for flag, scale, start, end in (
                    ('train', True, '2004-01', '2017-12'),
                    ('train', False, '2004-01', '2017-12'),
                    ('val', False, '2018-01', '2020-12'),
                    ('test', False, '2004-01', '2020-12'),
                ):
                    dataset = Dataset_Custom(
                        root_path=directory, data_path='city.csv', flag=flag,
                        size=[12, 0, 1], features='M', target='Amod2',
                        scale=scale, timeenc=0, freq='m',
                    )
                    actual = [tuple(map(int, dataset[i][3][-1, :2]))
                              for i in range(len(dataset))]
                    expected = [(p.year, p.month) for p in
                                pd.period_range(start, end, freq='M')]
                    assert actual == expected, (
                        flag, scale, len(actual), len(set(actual)), actual[:6]
                    )
                    assert dataset[0][0].shape == (12, 24)
                    assert dataset[0][1].shape == (1, 24)
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT / "baseline" / "st_mtm",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stmtm_scaling_and_target_inverse_transform(self):
        script = textwrap.dedent("""
            import tempfile
            from pathlib import Path
            from types import SimpleNamespace

            import numpy as np
            import pandas as pd
            from data_provider.data_factory import data_provider, data_provider_many

            with tempfile.TemporaryDirectory() as directory:
                dates = pd.date_range('2003-01-01', '2020-12-01', freq='MS')
                files = []
                for city in (0, 1):
                    frame = pd.DataFrame({'date': dates})
                    for i in range(23):
                        frame[f'feature_{i}'] = (np.arange(len(dates)) + city * 500) * (i + 1)
                    frame['Amod2'] = np.arange(len(dates)) / 10 + city * 10
                    frame.loc[180:, 'feature_0'] += 1_000_000
                    frame.loc[180:, 'Amod2'] += 1_000
                    file = f'city_{city}.csv'
                    frame.to_csv(Path(directory) / file, index=False)
                    files.append(file)

                args = SimpleNamespace(
                    data='Exchange', root_path=directory, data_path=files[0],
                    seq_len=12, label_len=0, pred_len=1, features='M',
                    target='Amod2', embed='fixed', freq='m', seasonal_patterns='Monthly',
                    batch_size=16, num_workers=0, task_name='pretrain',
                )
                raw = pd.read_csv(Path(directory) / files[0])
                original = raw.drop(columns='date').to_numpy()
                expected_mean = original[:180].mean(axis=0)
                expected_scale = original[:180].std(axis=0)
                reference = None
                for task in ('pretrain', 'finetune'):
                    args.task_name = task
                    for flag in ('train', 'val', 'test'):
                        dataset, _ = data_provider(args, flag)
                        assert dataset.scale
                        np.testing.assert_allclose(dataset.scaler.mean_, expected_mean)
                        np.testing.assert_allclose(dataset.scaler.scale_, expected_scale)
                        if reference is None:
                            reference = dataset.scaler.mean_.copy()
                        np.testing.assert_allclose(dataset.scaler.mean_, reference)
                        if flag == 'test':
                            transformed = dataset.data_y[180:181]
                            np.testing.assert_allclose(
                                dataset.inverse_transform(transformed), original[180:181]
                            )
                    combined, _ = data_provider_many(args, files, 'train')
                    assert all(dataset.scale for dataset in combined.datasets)

                args.features = 'MS'
                dataset, _ = data_provider(args, 'test')
                target = dataset.data_y[180:181, -1:]
                np.testing.assert_allclose(
                    dataset.inverse_transform(target), original[180:181, -1:]
                )
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT / "baseline" / "st_mtm",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == "__main__":
    unittest.main()
