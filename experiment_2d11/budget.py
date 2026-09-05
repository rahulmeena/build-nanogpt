"""Conservative measured projection and cumulative billing limits."""
import math
from .common import CE_SCHEDULE,HELLA_SCHEDULE,ENDPOINT


def project_remaining(measurements,elapsed_billed_seconds,completed=0,ce_done=(),hella_done=(),ceiling=86400):
    threes=ENDPOINT//32-completed//32
    twos=ENDPOINT-completed-threes
    training=twos*measurements['two_pass_seconds']+threes*measurements['three_pass_seconds']
    monitoring=sum(u not in ce_done for u in CE_SCHEDULE)*measurements['monitor_ce_seconds']
    hella=sum(u not in hella_done for u in HELLA_SCHEDULE)*measurements['h_hellaswag_seconds']
    final=measurements['h_fresh_ce_seconds']+measurements['g_fresh_ce_seconds']+measurements['g_hellaswag_seconds']
    saves=len([u for u in range(500,ENDPOINT,500) if u>completed])+1+int(completed==0)
    # The single bounded export worker overlaps subsequent GPU work. Simulate its
    # queue at HALF measured bandwidth; count CPU write/reopen time as blocking
    # even though the real writer is asynchronous. Required evaluations stay intact.
    write=measurements['checkpoint_write_seconds']
    transfer=measurements['checkpoint_bytes']/(measurements['download_bytes_per_second']*.5)
    gpu_clock=0.;export_clock=0.;events=[]
    # Conservatively reserve the most recent outstanding checkpoint on a revised forecast.
    if completed:export_clock=transfer
    for u in range(completed,ENDPOINT+1):
        if u>completed:gpu_clock+=measurements['three_pass_seconds'] if u%32==0 else measurements['two_pass_seconds']
        if u in CE_SCHEDULE and u not in ce_done:gpu_clock+=measurements['monitor_ce_seconds']
        if u in HELLA_SCHEDULE and u not in hella_done:gpu_clock+=measurements['h_hellaswag_seconds']
        if u==ENDPOINT:gpu_clock+=final
        if u==ENDPOINT or (u==0 and completed==0) or (u>completed and u%500==0):
            gpu_clock+=write
            export_clock=max(export_clock,gpu_clock)+transfer
            events.append(dict(update=u,file_ready_seconds=gpu_clock,export_finished_seconds=export_clock))
    tail=max(0.,export_clock-gpu_clock)
    exports=saves*(write+transfer)
    raw=gpu_clock+tail
    projected=raw*1.20+300
    available=ceiling-elapsed_billed_seconds-7200
    return dict(fits=projected<=available,remaining_projected_seconds=projected,
                remaining_allowance_less_two_hour_contingency=available,
                training_seconds=training,monitor_ce_seconds=monitoring,hellaswag_seconds=hella,
                final_evaluation_seconds=final,checkpoint_export_worker_seconds=exports,
                checkpoint_export_seconds=tail+saves*write,export_tail_seconds=tail,
                export_bandwidth_safety_fraction=.5,export_queue_events=events,
                writes_conservatively_counted_as_blocking=True,intermediate_exports_overlap_gpu_work=True,
                conservative_allowance_seconds=raw*.20+300,
                projected_remaining_gpu_hours=projected*4/3600,
                elapsed_billed_seconds=elapsed_billed_seconds,ceiling_seconds=ceiling,
                projected_g_four_gpu_training_seconds=ENDPOINT*measurements['g_update_seconds'],
                g_four_gpu_time_is_projection=True)


def retained_checkpoint_bytes(checkpoint_bytes,initial_bytes,transfer_probe_bytes=128<<20):
    # Seven permanent optimizer-boundary states, two rolling states, one in-flight snapshot/file.
    # Model-only initial snapshot is also frozen independently. Reserve 2 GiB for artifacts/code.
    return int(10*checkpoint_bytes+initial_bytes+transfer_probe_bytes+(2<<30))
