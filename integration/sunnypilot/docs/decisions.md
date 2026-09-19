# Sunnypilot NNSLR integration decisions

Baseline recorded for R00 of the runtime/reapply plan.

## Pinned baseline

- TRAIN_REPO source baseline: `cristianku/openpilot-nnslr-tools@9ffa00057905025c9fe6c8226664acc52330c32d`.
- Runtime fork: `cristianku/sunnypilot`.
- Runtime feature branch at bootstrap: `nn-speed-limit-vision`.
- Upstream source: `sunnypilot/sunnypilot:master`.
- Runtime/upstream commit observed for the first bootstrap: `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56`.

Every future reapply run must resolve the requested upstream ref again and freeze the resolved SHA for that run. The SHA above is a bootstrap baseline, not a permanent definition of "current master".

## Runtime naming

The runtime package and protocol names follow the v4 plan:

- package: `openpilot/sunnypilot/speed_vision/`
- daemon: `speedvisiond`
- public advisory state: `speedVisionState`
- optional co-located observation service: `speedVisionObservations`

Alternative names discussed earlier are not additional services.

## Isolation boundary

NNSLR is advisory only.

It must not:

- write `CarStateSP.speedLimit`;
- write or republish `LiveMapDataSP` as if Vision were map truth;
- add Vision to the operational `SpeedLimitResolver` or `SpeedLimitAssist`;
- emit CAN, cruise-button, acceleration, steering, radar, safety, or actuator requests;
- change the selected driving model/runner merely to make NNSLR available.

The existing CAR/MAP/SLA behavior is compared against the newly selected upstream baseline, not against an old historical branch.

## Chestnut

The inspected stock `modeld.py` states that only modeld can access Chestnut. A second NNSLR process must therefore never initialize Chestnut. A Chestnut backend, if later selected, must execute inside the existing owner and publish observations asynchronously to `speedvisiond`.

No target hardware backend is declared available by this document.

## Logging and privacy

NNSLR messaging is non-default-logged. Private diagnostics are opt-in, bounded, and separate from ordinary logger/uploader flows. Raw frames are not diagnostic output by default.

Route video, qlog/rlog, coordinates, annotations, training checkpoints, and private evaluation results remain outside Git.

## Artifact ownership

`openpilot-nnslr-tools` owns the canonical portable core, training/evaluation, model bundle metadata, and reapply tooling. The runtime repository consumes a generated core snapshot and immutable model artifacts; it does not maintain a second hand-edited core implementation.

The existing `openpilot/sunnypilot/neural_network_data` submodule is not modified as part of the bootstrap and is not assumed to be the final distribution channel for NNSLR artifacts.

## Reapply policy

A reapply produces a new candidate from a newly frozen upstream SHA. It does not force-update the stable runtime branch and does not deploy to the vehicle. The previous candidate remains identifiable for rollback.

Mechanical hook moves may be adapted by the reapply workflow with tests. Schema-binding conflicts, control-surface changes, accelerator ownership changes, or an inability to preserve isolation are blocking compatibility findings rather than automatic rewrites.

## Unknowns intentionally left open

The following require later evidence and are not guessed during source bootstrap:

- target runtime backend;
- stock-vs-native model runner on the device;
- stable Cap'n Proto binding chosen for the new messages;
- target-private diagnostics directory;
- on-device timing/thermal headroom;
- supported road-semantics capability level.
