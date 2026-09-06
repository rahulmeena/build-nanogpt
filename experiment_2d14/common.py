from pathlib import Path
from experiment_2d11.common import (sha256,identity,canonical,read_json,atomic_json,atomic_bytes,append_json,learning_rate,configure_cuda_determinism)
PACKAGE=Path(__file__).resolve().parent
REPO=PACKAGE.parent
FROZEN=PACKAGE/'frozen'
INIT_SHA='387b8b791ced54c2f1c0da65c47d00f35e8c3d1514897bad4c70db36db6c5c94'
H_SHA='559af751b9cceda5a4b08ef785d64f015cc4ce1978da7130a687f228f80a3244'
BASE_SHA='7680b0c094254f9b7bd11a9208d753867c9ce68f2efab01cb185934736988851'
H_MODEL_SHA='7d0f5d3da3dcc41a5c9e83c4d3a94fa4aa3caa7be226d1d92521d3c2f0f7548a'
VAL_SHA='8e06151653328dbbd1a225bf0ab3ea902c561564c76d9fc2dc6278be8f754c0f'
SCHEDULE=(0,250,500,750,1000)
FULL_INIT_SHA='c1ad032bfc2fccd4fdabae0f0c4064f856b75ae8037e2373a1cc02f8216d6386'
CONDITIONS=('H_ON','L_LOCAL','R_ON','H_ALL_OFF','R_ALL_OFF')
def pass_count(u):
    assert 1<=u<=1000
    return 3 if u%32==0 else 2
def accounting(u):
    assert isinstance(u,int) and 0<=u<=1000
    passes=2*u+u//32
    return dict(completed_updates=u,logical_targets=524288*u,two_pass_updates=u-u//32,
                three_pass_updates=u//32,backbone_pass_equivalent_updates=passes,
                backbone_pass_targets=524288*passes,actual_ce_target_evaluations=524288*passes,
                nonzero_weight_ce_targets=524288*(u+u//32),diagnostic_only_ce1_targets=524288*u,
                mean_backbone_passes=passes/u if u else None)
