import os
import yaml
import random
import mlflow

#ディレクトリの取得
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

#パラメータロード
with open(os.path.join(FILE_DIR, "./config/config.yml")) as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    # mlrunsの準備
    client = mlflow.tracking.MlflowClient()

    try:
        # 実験フォルダをmlruns内に新規作成する場合はGCSがArtifactsの保存先になるように指定する
        exp_id = client.create_experiment(config['experiment_name'], artifact_location=f"gs://{config['bucket_name']}/artifacts")
    except:
        # すでにmlruns内に同じexperiment_nameのフォルダが存在する場合、そのArtifactの保存先をGCSに強制的に書き換える
        exp_id = client.get_experiment_by_name(config['experiment_name']).experiment_id
        filepath = f"{FILE_DIR}/mlruns/{exp_id}/meta.yaml"
        with open(filepath) as file:
            meta = yaml.safe_load(file)
        meta["artifact_location"] = f"gs://{config['bucket_name']}/artifacts"

        with open(filepath, 'w') as file:
            yaml.dump(meta, file, default_flow_style=False)
            
    # MLprojectをgithubから持ってきて実行する
    mlflow.projects.run(config['git_uri'], entry_point='main', experiment_name=config['experiment_name'], use_conda=False)