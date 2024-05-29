from data.zsre import MendTrainDataset, MENDQADataset
from data.pararel import PARARELTrainDataset, PARARELQADataset

from evaluate.eval_utils_zsre import compute_rewrite_quality_zsre

TRAIN_DS_DICT = {
    "zsre": MendTrainDataset, 
    "pararel": PARARELTrainDataset,
}
DS_DICT = {
    "zsre": MENDQADataset, 
    "pararel": PARARELQADataset,
}
EVAL_METHOD_DICT = {
    "zsre": compute_rewrite_quality_zsre, 
    "pararel": compute_rewrite_quality_zsre,
}