# The implementation of [DASR]
## Version info: pytorch 2.0.0, python 3.9

### Steps

Modify：
Define your path in _configs/dasr.yaml_

Choose:
Choose parameter in _configs/dasr.yaml_

|  parameter   | explanation  |
|  ----  | ----  |
| _detect_mode_  | Run single pic or dataset |
| _feature_save_  | Whether to save feature data |
| _plt_mode_  | Whether to show / save result by plt |
| _display_mode_  | Display result by bounding box or response map |
| _dataset_  | Run which dataset |
| _transform_mode_  | Normalize by different mean and std |
| _model_select_  | Process query or reference |
| _use_nms_  | Whether to use nms |
| _use_DASR\*_  | Whether to use dasr* |
| _use_rank_  | Whether to use rank |
| _backbone_  | Choose pre-train parameter for backbone |
| _feature_map_pooling_  | Whether to use pooling on feature map |



Run:
```pythobn
python main.py 
```

### Reference
```markdown
@article{xiao2022deeply,
  title={Deeply activated salient region for instance search},
  author={Xiao, Hui-Chu and Zhao, Wan-Lei and Lin, Jie and Hong, Yi-Geng and Ngo, Chong-Wah},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={18},
  number={3s},
  pages={1--19},
  year={2022},
  publisher={ACM New York, NY}
}
```

### Publisher 
**Yi-Bo Miao Xiamen University**