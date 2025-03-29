# HOGraspNet

HOGraspNetは、ECCV 2024で発表された「Dense Hand-Object(HO) GraspNet with Full Grasping Taxonomy and Dynamics」におけるhand meshからgrasp taxonomyを分類するmodelです.

プロジェクトページ: [HOGraspNet](https://hograspnet2024.github.io/)

## 各ファイルの説明
- `src/train/train.py`: trainする
- `vis_mesh.py`: datasetのhand mesh, object mesh, contact mapを可視化する
- `inference.py`: trainしたmodelをloadしてinferenceする
- `config/config.json`: modelのconfig