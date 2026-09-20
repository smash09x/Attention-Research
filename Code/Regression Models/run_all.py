"Runs every script in order and saves each output to results/. Run: python run_all.py --data ../../Data/CSVs"

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(name, script, args, results):
    print(f"running {name} ...")
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, cwd=HERE)
    (results / f"{name}.txt").write_text(done.stdout + done.stderr)
    if done.returncode:
        print(done.stderr[-800:])
        raise SystemExit(f"{name} failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument('--data', default='../../Data/CSVs')
    args = parser.parse_args()

    data = Path(args.data).resolve()
    results = HERE / 'results'
    results.mkdir(exist_ok=True)

    primary = ['--trends', str(data / 'trends.csv'),
               '--strat', str(data / 'keywords_stratified.csv'),
               '--youtube', str(data / 'youtube_volume.csv')]
    age = ['--trends', str(data / 'trends_age.csv'),
           '--strat', str(data / 'keywords_stratified_age.csv'),
           '--youtube', str(data / 'youtube_volume_age.csv')]
    common = primary + ['--trends-age', str(data / 'trends_age.csv'),
                        '--strat-age', str(data / 'keywords_stratified_age.csv'),
                        '--youtube-age', str(data / 'youtube_volume_age.csv')]

    for model in ('A', 'B', 'C'):
        run(f"model_{model}_primary", 'model_main.py', [model] + primary, results)
        run(f"model_{model}_age", 'model_main.py', [model] + age, results)
    run('tables', 'regression_tables.py', ['all'] + common, results)
    run('robustness', 'robustness_checks.py', common, results)
    run('survival', 'survival_models.py', common, results)
    run('extras', 'paper_extras.py', common, results)
    run('figure_1', 'figures.py', common + ['--out', str(results / 'figure1.png')], results)
    print("done, outputs in results/")
