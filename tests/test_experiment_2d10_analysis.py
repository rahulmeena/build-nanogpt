"""Strict boundaries and independent verification of the saved paired analysis."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import experiment_2d10_analysis as a

def test_strict_decision_boundaries():
    assert not a.flags([0,.001])['positive']
    assert not a.flags([.0001,.001])['beyond_margin']
    assert not a.flags([-.001,0])['negative']
    assert not a.flags([-.001,-.0001])['material_harm']
    assert not a.flags([-.0001,.00005])['practical_equivalence']
    assert not a.flags([-.00005,.0001])['practical_equivalence']
    assert not a.flags([-.0001,.00005])['second_condition_noninferiority']
    assert a.flags([-.00009,.00009])['practical_equivalence']

def test_saved_primary_pairing_and_intervals():
    values=np.load(a.RESULT/'PAIRED_SEQUENCE_LOSSES.npz')
    draws=np.load(a.ARCHIVE/'BOOTSTRAP_MEANS.npy')
    stats=a.read(a.RESULT/'PAIRED_BOOTSTRAP.json')
    assert draws.shape==(50000,6)
    indices=np.random.default_rng(20260910).integers(0,4096,size=(128,4096))
    for i,(name,first,second) in enumerate(a.CONTRASTS):
        difference=values[first]-values[second]
        np.testing.assert_allclose(draws[:128,i],difference[indices].mean(1),rtol=0,atol=1e-15)
        row=stats['contrasts'][name]
        assert abs(row['mean']-difference.mean())<1e-15
        np.testing.assert_allclose(np.quantile(draws[:,i],[.025,.975]),row['raw_95_ci'],rtol=0,atol=1e-15)
        if i<3:
            expected=np.quantile(draws[:,i],np.asarray(a.PRIMARY_Q)/100)
            np.testing.assert_allclose(expected,row['adjusted_98_333333_ci'],rtol=0,atol=1e-15)
            assert a.flags(expected)==row['adjusted_flags']
