"""Pure serial state machine. Exceptions and budgets are never stop triggers."""
from .common import ARMS,MILESTONES,SCHEDULE,CONDITIONS

def required_labels(arm):
    monitors=[f'{arm}_monitor_u{u:05d}' for u in SCHEDULE]
    conditions=('L_LOCAL','H_ON','H_ALL_OFF') if arm=='L_nf4' else ('R_ON','R_ALL_OFF')
    return monitors+[f'{c}_u{u:05d}' for u in MILESTONES for c in conditions]

def arm_ready(state,arm):
    a=state.get(arm,{})
    return (a.get('updates')==5000 and a.get('gpu_complete') is True and
            set(a.get('evaluations',[]))==set(required_labels(arm)) and
            a.get('exports_verified') is True)

def complete(state):
    return all(arm_ready(state,a) for a in ARMS) and state.get('joint_identity_coverage_verified') is True and state.get('remaining_gpu_work')==0

def next_stage(state):
    if not state.get('local_ready'):return 'LOCAL_READY_PENDING'
    if not state.get('both_preflights'):return 'BOTH_ARMS_PREFLIGHT_PENDING'
    if not arm_ready(state,'L_nf4'):return 'L_TRAIN_AND_SCORE'
    if not state.get('R_original_reset_verified'):return 'R_RESET_TO_ORIGINAL_INITIAL_STATE'
    if not arm_ready(state,'R_nf4'):return 'R_TRAIN_AND_SCORE'
    return 'STOP_ASSIGNED_POD' if complete(state) else 'JOINT_GPU_OUTPUTS_VERIFY'

def maybe_stop(state,stop_callback):
    if complete(state):stop_callback();return True
    return False
