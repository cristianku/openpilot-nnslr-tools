---
title: T0 audit — NN Vision Speed Limit (NNSLR)
type: audit
repos: [nn-speed-limit-vision, openpilot-nnslr-tools, openpilot_scripts]
updated: 2026-09-18
status: T0 complete, pending review (G0 not yet accepted)
---

# T0 — Read-only audit and first-increment decision

Audit eseguito il **2026-09-18** su richiesta di Cristian ("follow the steps described").
Scope: **solo lettura** sui repository + **scrittura** di questi due documenti in `TRAIN_REPO`.
Nessun checkout modificato, nessun push, nessun device, nessun GPU host toccato.

Mappatura dei root (decisa da Cristian in sessione, differisce dal piano originale):

| Simbolo piano | Repository reale | Note |
|---|---|---|
| `TRAIN_REPO` | `/Users/cristianku/GitHub/COMMA.AI/CRISTIANKU/openpilot-nnslr-tools` | **Esiste già** (non proposta da creare): remote `git@github.com:cristianku/openpilot-nnslr-tools.git`, branch `main`, `969265d` "first commit" (solo `README.md` vuoto). Il piano `NN_VISION_SPEED_LIMIT_ALL_IN_ONE_v4.md` è **untracked** in questo repo. |
| `RUNTIME_REPO` | `/Users/cristianku/GitHub/COMMA.AI/CRISTIANKU/nn-speed-limit-vision` | Fork di **sunnypilot/master** (remote `git@github.com:cristianku/sunnypilot.git`), branch `master`, HEAD `a5f44653d7` (tag `staging/2026.003.000/2026.09.16-4892`), working tree **pulita**. È un checkout persistente, non generato dalle skill `op-setup-*`. |
| Knowledge hub | `/Users/cristianku/GitHub/COMMA.AI/CRISTIANKU/openpilot_scripts` | Branch `main`, `4e3654a`, pulita. |

> **Divergenza dal piano:** il piano (sez. 2.1 e 16) registra come riferimento `cristianku/openpilot` su branch `psa-torque-sunny-testing`, commit storico `4bfe6f87f5d9e6d8bd2b618cbcfa9b11c2d812d9`. Cristian ha indicato `nn-speed-limit-vision` (fork di `sunnypilot/master`) come sede dei cambiamenti. Il commit `4bfe6f8` **non esiste** in questo fork (`git cat-file -t` → fatal), coerente con la diversa linea di fork. Tutte le osservazioni qui sotto sono state **ri-verificate sul checkout reale** e non si basano sul commit storico.

## 1. Repository e workspace (evidenze)

### TRAIN_REPO (`openpilot-nnslr-tools`)
- `git log -1`: `969265d first commit` — contiene solo `README.md` (1 riga).
- `git status --short`: untracked `NN_VISION_SPEED_LIMIT_ALL_IN_ONE_v4.md` (il piano rev 4).
- Remote: `origin git@github.com:cristianku/openpilot-nnslr-tools.git` (fetch/push).
- **Conclusione:** il repository remoto **esiste già** (creato da Cristian). Nome e visibilità: **decisi**, non più in sospeso. La visibilità GitHub (public/private) non è verificata in locale — da confermare da Cristian.

### RUNTIME_REPO (`nn-speed-limit-vision`)
- Branch `master`, HEAD `a5f44653d7`, tag `staging/2026.003.000/2026.09.16-4892`, working tree pulita, nessun untracked.
- `git submodule status` (tutti **non inizializzati**, prefisso `-`):
  | Submodule | SHA |
  |---|---|
  | `msgq_repo` | `e7396e76dadbb49e374d4b664ff6bbb43a39bcb0` |
  | `opendbc_repo` | `f95f996f5917dcbbf2e32fe51b606a24cf836af6` |
  | `openpilot/sunnypilot/neural_network_data` | `03cac2d30e111e0689c0429cb8c1fe6cb5a905af` |
  | `panda` | `74a0adced421e8b7acd728d0f9988ce225423f13` |
  | `rednose_repo` | `28d4a7f69e80e1c3e0d24ca0733d7daeaeade3d0` |
  | `teleoprtc_repo` | `1aa8fc433bef1519a95c0700c96258c3be6dfb34` |
  | `tinygrad_repo` | `f6fc4e3f2c3db5fae1e19cbfbc3ad9fc579a12ae` |
- Remoti dei submoduli (`.gitmodules`): `opendbc` → `sunnypilot/opendbc`, `panda` → `sunnyhaibin/panda`, `tinygrad` → `sunnypilot/tinygrad`, NN data → `sunnypilot/neural-network-data` (conferma linea sunnypilot, non commaai).
- **Nessuno dei submoduli è inizializzato localmente** → questo checkout non è costruibile così com'è; è una fonte di consultazione. Per test runtime servirà un worktree separato con submoduli (o le skill `op-setup-psa-torque*` se il runtime diventa un requisito — **non** per ora: T1–T8 non toccano RUNTIME_REPO).

### openpilot_scripts
- `main` @ `4e3654a`, pulita. Wiki operativa letta: `wiki/index.md`, `wiki/entities/psa-3008-can-reverse-engineering.md` (finding TSR, §6).

## 2. Verifica delle osservazioni del snapshot (sez. 2.1 del piano)

Tutte ri-verificate su `nn-speed-limit-vision@master`. Layout confermato: i file sono sotto `openpilot/...` (es. `openpilot/cereal/services.py`), coerente con il piano.

| # | Osservazione registrata nel piano | Verificato su | Esito |
|---|---|---|---|
| S2 | `ChestnutState` con commento di single-owner | `openpilot/selfdrive/modeld/modeld.py:78-79` (`class ChestnutState:` + `# only modeld can access chestnut`) | **Confermato** |
| S3 | `process_config.py` sceglie `modeld` (stock) vs `modeld_tinygrad` | `openpilot/system/manager/process_config.py:86-91` (`is_tinygrad_model`/`is_stock_model` via `get_active_model_runner`), `:124` (`PythonProcess("modeld", ..., and_(only_onroad, is_stock_model))`), `:172` (`NativeProcess("modeld_tinygrad", "openpilot/sunnypilot/modeld_v2", ["./modeld"], and_(only_onroad, is_tinygrad_model))`) | **Confermato, con dettaglio in più** (vedi §3) |
| S4 | Resolver legge `carStateSP.speedLimit` + `liveMapDataSP`; policy combined = minimo positivo | `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py`: `:115` (`sm['carStateSP'].speedLimit`), `:130-131` (`liveMapDataSP.speedLimit/speedLimitAhead` + valid flag), `:158-166` (`_get_source_solution_according_to_policy`, combined = `min(...)`), `:128-129` (watchpoint time-domain: `time.monotonic() - gps_data.unixTimestampMillis * 1e-3` — **presente**, §2.3) | **Confermato** |
| S5 | `speed_limit_assist.py` = stato Assist + target speed | `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py` (415 righe; `:65` `self.enabled = ... SpeedLimitMode.assist`) | **Confermato** |
| S6 | Enum `Source`: `none @0`, `car @1`, `map @2` | `openpilot/cereal/custom.capnp:283-287` (`enum Source { none @0; car @1; map @2; }`) | **Confermato** |
| S7 | Servizi narrow/wide + decimazioni | `openpilot/cereal/services.py`: `narrowRoadEncodeIdx (False, 20., 1)`, `wideRoadEncodeIdx (False, 20., 1)`, `narrowRoadCameraState (True, 20., 20)`, `wideRoadCameraState (True, 20., 20)` | **Confermato** |
| S8 | `tools/lib/route.py`: `fcamera.hevc`/`ecamera.hevc`/`qcamera.ts`/`rlog`/`qlog` | `openpilot/tools/lib/route.py:15-19` | **Confermato** |
| S9 | UI entry point Python | `openpilot/selfdrive/ui/ui.py`: `gui_app.init_window("UI")`, `MainLayout()` (BIG_UI) oppure `MiciMainLayout()`, `ui_state.update()` nel loop | **Confermato** |
| S10 | Submodule `opendbc_repo`, `msgq_repo`, `tinygrad_repo` | `.gitmodules` (tabella §1) | **Confermato** (+`panda`, `rednose_repo`, `teleoprtc_repo`, `neural_network_data`) |

**Osservazioni nuove emerse dall'audit (non nel piano):**

1. **Runner selection è param-driven, non compile-time.** `openpilot/sunnypilot/models/helpers.py:178-194` (`get_active_model_runner`): legge il bundle attivo da params (validato contro il catalogo da `models_manager`, solo offroad), default `Runner.stock` se nessun bundle; enum `custom.ModelManagerSP.Runner { snpe @0; tinygrad @1; stock @2 }` (`custom.capnp:171-175`). Il runner attivo **sul comma non è determinabile da qui** (param `ModelRunnerTypeCache`/bundle attivo vivono nel device). Impatto su T9: la scelta Path A/B dovrà usare l'evidenza del device (autorizzata separatamente).
2. **`carStateSP.speedLimit` non ha un produttore Python in questo checkout.** `grep -rn "carStateSP\.speedLimit ="` → nessun assegnamento; gli unici consumer sono il resolver (`:115`) e `mapd/live_map_data/debug.py`. Coerente con il finding del knowledge hub: **il TSR della 3008 non è sui bus CAN tappati** (vedi §6). Conseguenza: nella 3008 il source `car` del resolver resterà 0 e il limite operativo è **map-only**. Vision non cambia questo: è un nuovo source advisory, non un alimentatore del `car` source.
3. **Il producer del map source è `mapd`** (processo native `openpilot/sunnypilot/mapd`, `process_config.py:180`): `openpilot/sunnypilot/mapd/live_map_data/base_map_data.py:34-57` (`get_current_speed_limit`, `speedLimitValid`, `speedLimitAhead`, `speedLimitAheadDistance`).
4. **UI SLA:** widget settings `openpilot/selfdrive/ui/sunnypilot/layouts/settings/cruise_sub_layouts/speed_limit_settings.py`; widget onroad `openpilot/selfdrive/ui/sunnypilot/onroad/speed_limit.py` (328 righe, `:193-201` gating per `SpeedLimitMode`); param `SpeedLimitMode` (persistente, default "1") in `openpilot/common/params_keys.h:278`. Il badge Camera di T12 dovrà seguire la convenzione dei layout sunnypilot (MainLayout vs MiciMainLayout in `ui.py`).
5. **Uploader (privacy):** due pipeline distinte: `openpilot/system/loggerd/uploader.py` (comma API `v1.4/<dongle>/upload_url/`, gated da `OnroadUploads`) e `openpilot/sunnypilot/sunnylink/uploader.py` (Sunnylink, `xattr user.sunny.upload`, `MAX_UPLOAD_SIZES` qlog/qcam). Entrambe leggono da `Paths` (dati device). Le diagnostiche private di `speedvisiond` (T9/T10) **non** devono cadere in queste radici; da audit in T9 con i percorsi reali.
6. **Test SLA esistenti** (per riuso in T7/T10): `openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_resolver.py` (6 test) e `test_speed_limit_assist.py` (24 test), stile `OpenpilotTestCase` + mocker. **Comando di regressione SLA da riusare in ogni task di integrazione:**
   ```bash
   # da RUNTIME_REPO, con submoduli inizializzati e ambiente di test (da T7, non eseguibile ora: submoduli non inizializzati)
   python -m pytest openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/tests/ -q
   ```

## 3. Chestnut e scheduling (per T9, oggi solo facts)

- `modeld.py:245-263`: `chestnut_present() and chestnut_compiled()` → handshake `chestnutState` (10 Hz, decimato 10 in `services.py:28`), `ChestnutActive`/`ChestnutLoading` in params. `:313`: se Chestnut attivo, `small_model = ... if model is None or CHESTNUT else None` → **un solo owner GPU** (modeld), confermato il vincolo di single-owner del piano.
- `services.py`: `modelV2 (True, 20., None, QueueSize.BIG)`, `drivingModelData (True, 20., 10)`, `modelDataV2SP (True, 20., None, BIG)` — la proposta del piano di **non** sovraccaricare `modelV2` resta valida.
- Margini GPU, termica e feasibility: **non stabiliti** (richiede autorizzazione esplicita + device; G4).

## 4. Clock domains e allineamento frame/log (per T2)

- `narrowRoadEncodeIdx`/`wideRoadEncodeIdx` hanno `timestampSof`/`timestampEof` (uso confermato in `openpilot/tools/jotpluggler/layouts/camera-timings.json` e `openpilot/selfdrive/test/process_replay/vision_meta.py:12-13`, che mappa `narrowRoadCameraState`→`VisionStreamType.VISION_STREAM_NARROW_ROAD`, DT_MDL).
- `rlog`/`qlog` registrano gli encode-idx a 20 Hz (`services.py`); `qlog` è decimato ma **contiene** gli encode-idx (`should_log=False` ma "EncodeIdx packets will still be in the log" — comment in `services.py:19`, e verificato empiricamente: vedi §5).
- Watchpoint time-domain (S4): `unixTimestampMillis` vs `time.monotonic()` nel resolver — **presente** (`speed_limit_resolver.py:128-129`, `:139`). Non copiare quell'aritmetica in Vision; non refactorizzare il resolver in questa feature branch (vincolo del piano §2.3).
- **Da stabilire in T2:** semantica esatta di SOF/EOF per questa versione, join frame↔log, gestione GOP parziali. Il piano §5.3 resta la procedura.

## 5. Inventario dati locale (per T2/T3 e proposta clip)

**`openpilot_scripts/comma_logs/`**: **106 segmenti**, file solo `.zst` (rlog.qlog) + 20 CSV già convertiti.

- **Nessun video** (`fcamera.hevc`/`ecamera.hevc`/`qcamera.ts` assenti in tutti i segmenti — le route sono state scaricate con `--no-video`).
- 5 route: `00000050--eabf0e8324` (4 seg), `00000051--3689ce6893` (22 seg), `00000058--c9356c573d` (27 seg), `00000070--97bd1ebf25` (25 seg), `00000082--cb04e38b5c` (8 seg).
- CSV convertiti (`csv/`): solo per `00000050--eabf0e8324`, `00000051--3689ce6893`, `00000070--97bd1ebf25` — e solo `carControl,carState,controlsState,selfdriveState,logMessage,errorLogMessage`. **Niente `liveMapDataSP` nei CSV** (ma presente nei log grezzi, sotto).
- Verifica empirica (`log_to_csv.py --list --qlog`, `CEREAL_DIR=.../nn-speed-limit-vision/openpilot/cereal`), route `00000082--cb04e38b5c`: `narrowRoadEncodeIdx` **8572**, `wideRoadEncodeIdx` **8572**, `qNarrowRoadEncodeIdx` 8577, `liveMapDataSP` **430**, `gpsLocationExternal` **428**, `carStateSP` 4192, `narrowRoadCameraState`/`wideRoadCameraState` 430 (qlog decimato 20). → i join per T2 sono fattibili sui log già presenti **per il metadato**; il **video va recuperato** (vedi sotto).

**Implicazioni per T2 (blocco noto, non bloccante per T1):**
1. **Il video non è in locale.** Serve: (a) download da device delle route scelte (`OP-download-comma-logs` **senza** `--no-video`, con autorizzazione), oppure (b) nuove route raccolte apposta con video (raccolta sicura, pianificata, §5.5 del piano). Fino ad allora T2 si sviluppa con **fixture sintetiche** + il contratto di alignment, e l'allineamento reale resta *not established*.
2. `NNSLR_DATA_ROOT`: da creare **fuori** dai due repo (proposta: `/Users/cristianku/GitHub/COMMA.AI/CRISTIANKU/speed-vision-data/` — da confermare da Cristian).
3. Le 5 route locali sono guide PSA in Svizzera: utili per il **metadato** (encode-idx, GPS, `liveMapDataSP`), ma **non** ancora per la percezione (nessun frame).

## 6. Finding contestuale (knowledge hub)

Da `openpilot_scripts/wiki/entities/psa-3008-can-reverse-engineering.md` (drives 2026-07-02, dongle `6616faac453a3064`, due passaggi di segnaletica 60 km/h confermati su video):

- **Negativo principale:** il limite riconosciuto (il "60" in cruscotto) **non compare sui bus 0/1/2** tappati dal comma (nessun campo stabile a entrambi gli eventi). Ipotesi: fusione camera+nav dentro NAC/BSI su IS/infotainment CAN, non visibile dal connettore ADAS. Per catturarlo servirebbe un logger CAN su OBD/cruscotto.
- Conseguenza operativa già in atto: su questa vettura `carStateSP.speedLimit` non è alimentabile da CAN e lo SLA resta **map-only** (vedi anche `concepts/longitudinal-control.md`, sezione SLC).
- **Pertinenza per NNSLR:** rafforza la motivazione del progetto (una fonte di limite basata sulla camera, advisory, senza toccare il control path), e conferma che **non** esistono ground truth TSR locali nei log CAN per addestrare o validare il road-context: l'annotation umana resterà l'unica fonte di verità.

## 7. Hardware / accesso

- **Device comma:** non contattato, non ispezionato (nessuna autorizzazione in questa sessione). Stato installato, runner attivo, Chestnut: **unknown**.
- **GPU host (V100):** non ispezionato (nessuna autorizzazione). Nessun benchmark eseguito.
- **Mac locale** (questo ambiente): x86_64 darwin; l'ambiente Python openpilot non si builda qui (nota in `log_to_csv.py`); i test TRAIN_REPO CPU devono girare in un ambiente isolato (venv/uv), **non** dentro il checkout openpilot.

## 8. Proposta: primi 10 clip e test T1–T3

I clip non sono ancora identificabili con nomi/timestamp (nessun video locale; i passaggi di segnaletica noti dal wiki sono su route **non** presenti in `comma_logs/`: `00000019--131cf7274b` seg 3, `00000022--b31b936f36` seg 2). **Proposta operativa in due fasi:**

**Fase 1 (orizzontale immediata, senza device):**
1. Fixture sintetica di "route" (video sintetico + encode-idx/GPS sintetici) per T2: segmento mancante, frame ID duplicato, conteggi video/log divergenti (già richiesti dal piano T2).
2. Fixture di annotazioni con box fuori range, valore non leggibile, panel associato/assente, per T3.
3. Caso "commute connessa" (un solo gruppo) per il report di split impossibile 70/15/15 (T3).
4. Golden image per preprocessing (letterbox 640/960, crop 96/128, coordinate originali) (T4).

**Fase 2 (dopo recupero video, autorizzazione separata):**
5-14. Dieci clip di approach+passage da selezionare dalle route locali **con video** (download `--no-video` **no**), criteri: segno numerico statico visibile inurno, transizione `liveMapDataSP.speedLimit` o cambio di limite di mappa nei ±30 s (già estratti dai qlog locali come **suggerimento di clip**, non ground truth), copertura di: bordo strada, svincolo/parallel, autostrada (se presenti), LED/pannello integrativo come casi negativi. La selezione finale richiede l'ispezione visiva dei frame (T2 step 6 del piano).

## 9. Gate e capacità

| Gate | Stato | Note |
|---|---|---|
| G0 — Audit accettato | **In sospesa review** (questo documento) | Nomi/root decisi da Cristian in sessione; visibilità GitHub da confermare; nessun remote creato da T0 |
| G1 — Data trustworthy | **Not established** | Niente video locale; l'allineamento reale è in attesa di dataset |
| G2 — Perception useful | **Not established** | Nessun training eseguito |
| G3 — Interpretation established | **Not established** | |
| G4 — Runtime feasible | **Not established** | Runner device unknown; richiede autorizzazione + misurazioni |
| G5–G7 | **Not established** | |

**Blocco più vicino:** G1 dipende dal recupero/raccolta video (Fase 2). Non blocca T1 (contratti + fixture) e gran parte di T2 (logica di alignment su fixture).

## 10. Prima proposta di implementazione offline (T1) e criteri di accettazione

Scope T1 (solo `TRAIN_REPO`, CPU, nessuna dipendenza da openpilot/device):

1. Packaging `pyproject.toml` + layout della sezione 12.2 (solo i file del task T1 — **niente** stub vuoti per script successivi).
2. `src/speed_vision_core/types.py` — tipi e invarianti della sezione 7 (presenza esplicita, domini di clock, `value_kph: int | None`, mai zero-reattribuito).
3. `src/nnslr_tools/cli.py` — CLI `nnslr` con subcomandi reali per ciò che esiste (es. `version`, `validate-batch` su fixture).
4. `tests/core/test_contracts.py`, `tests/cli/test_entrypoints.py`, `tests/cli/test_import_isolation.py` + `tests/fixtures/` (osservazioni valide, valore unknown, score non-finiti, box fuori range, session mismatch, duplicati, evidence stale).
5. `docs/plan.md` = la specifica completa (questo piano rev 4), `README.md` quickstart CPU/sintetico, `.gitignore` per dati privati, `AGENTS.md`.

**Accettazione T1:** clone pulito → quickstart CPU funziona senza comma, senza openpilot, senza GPU; tutti i casi malformed hanno reason code deterministici; `import speed_vision_core` non importa CUDA/torch/messaging; round-trip serialization preserva assenza/unknown.

**Ordine proposto:** T1 → T2 (fixture) → T3 (fixture) → [pausa per decisione video/device] → T4 (richiede autorizzazione V100). T6 (core temporale) è fattibile in parallelo dopo T1 su fixture pure, se approvata.
