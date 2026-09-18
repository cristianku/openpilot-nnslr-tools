---
title: Decision records — NN Vision Speed Limit (NNSLR)
type: decisions
repos: [openpilot-nnslr-tools, nn-speed-limit-vision]
updated: 2026-09-18
status: D1–D4 chiuse (approvate in review 2026-09-18); D5 video-first approvata (esecuzione T2+); D7 T1 APPROVATA; D8 architettura VISION registrata; D6 aperta
---

# Decision records

Registrazione delle scelte con alternative, evidenza, opzione scelta, dominio supportato e condizione di riesame. Le decisioni di runtime, road-attribution e passage **non** possono essere giustificate solo da una demo riuscita (vincolo piano §15).

## D1 — Sedesede dei tre repository

- **Alternativa A (piano rev 4 originale):** training repo *da creare* (`cristianku/openpilot-nnslr-tools`, nome in proposta); runtime repo `cristianku/openpilot` su `psa-torque-sunny-testing`.
- **Alternativa B (scelta da Cristian, sessione 2026-09-18):** training in `openpilot-nnslr-tools` (fork **già esistente**, `main@969265d`, remote già su GitHub); runtime integration in `nn-speed-limit-vision` (**fork di sunnypilot/master**, `master@a5f44653d7`, tree pulita); knowledge hub `openpilot_scripts` (sola lettura per questo progetto).
- **Evidenza:** stato git ispezionato in `audit.md` §1; commit storico del piano `4bfe6f8` non presente nel fork (linea di fork diversa).
- **Scelta:** **B**.
- **Conseguenze:** il piano (sez. 2.1, 12.5, 16) va letto con `RUNTIME_REPO = nn-speed-limit-vision`. Le osservazioni del snapshot sono state ri-verificate su questo checkout (audit §2) — nessuna è risultata falsa, ma la runner-selection è param-driven (audit §2.3#1) e il producer `carStateSP.speedLimit` non esiste nel tree (audit §2.3#2).
- **Riesame:** se Cristian vorrà integrare nel fork `cristianku/openpilot` (linea `psa-torque-sunny*`) invece che in `nn-speed-limit-vision`, serve un nuovo T0 sul checkout giusto.

## D2 — Naming, ownership, visibilità

- **Fatto:** `cristianku/openpilot-nnslr-tools` **esiste** (remote SSH, creato da Cristian). Nome definitivo, owner Cristian.
- **Decisione (approvata in review 2026-09-18):** visibilità **PUBLIC**. Il progetto è pensato per essere riproducibile e vive nell'ecosistema openpilot/sunnypilot; training/eval/export restano indipendenti dal fork runtime.
- **Record per Appendix B:**
  ```text
  repository: cristianku/openpilot-nnslr-tools
  visibility: public
  ```
- **Operazioni remote:** nessun push in T1 (solo commit locali); la pubblicazione/push resta un'operazione separatamente autorizzata.

## D3 — Branch e worktree per l'integrazione runtime

- **Vincolo piano:** mai toccare il checkout usato per guidare senza permesso separato; integrazione in worktree/branch isolato, proposta `feat/vision-speed-limit-shadow`.
- **Fatto:** `nn-speed-limit-vision@master` è un checkout **persistente** (non generato dalle skill setup) e al momento è l'unico checkout runtime del workspace; i submoduli non sono inizializzati.
- **Scelta (proposta, da approvare):** T1–T8 **non** toccano RUNTIME_REPO. Al primo task runtime (T7 test di isolamento / T9) si crea un worktree separato (es. `../nn-speed-limit-vision-vision-wt` su branch `feat/vision-speed-limit-shadow`) **con submoduli inizializzati** — operazione da autorizzare esplicitamente in quel momento. Fino ad allora il comando di regressione SLA (audit §2.3#6) resta documentato ma non eseguita.
- **Riesame:** alla creazione del worktree.

## D4 — `NNSLR_DATA_ROOT`

- **Vincolo piano:** radice dati privata **fuori** da entrambi i repo, non committata, non condivisa via Git/CI.
- **Decisione (approvata in review 2026-09-18):** `/Users/cristianku/GitHub/COMMA.AI/CRISTIANKU/speed-vision-data/`, con layout del piano §5.1 (`raw/`, `derived/`, `manifests/`, `private/location-index.json`, `runs/`, `evaluation/`).
- **Vincolo di implementazione:** il percorso **non** va hardcodato nell'applicazione; il codice lo legge da variabile d'ambiente `NNSLR_DATA_ROOT` (o configurazione). I documenti ed esempi usano la forma generica `NNSLR_DATA_ROOT=/path/to/speed-vision-data`.
- **Nota:** `openpilot_scripts/comma_logs/` resta la radice di **download** dei log (convenzione esistente delle skill); il video scaricato andrà in `NNSLR_DATA_ROOT/raw/` (o symlink esplicito — da decidere in T2, mai un bind silenzioso).

## D5 — Fonte video per T2 (APPROVATA in review 2026-09-18: video-first)

- **Fatto:** 106 segmenti locali, **solo** rlog/qlog, zero video (scaricati `--no-video`). I qlog contengono `narrowRoadEncodeIdx`/`wideRoadEncodeIdx`/`liveMapDataSP`/`gpsLocationExternal` (audit §5).
- **Decisione (approvata in review 2026-09-18):** **recuperare per prima il video dalle route esistenti sul comma**, se ancora disponibile, prima di registrare nuove guide. Priorità:
  1. audit read-only del comma;
  2. identificare quali dei 106 segmenti hanno ancora il video della camera road;
  3. scaricare solo i file video corrispondenti;
  4. join frame video ↔ metadati qlog/rlog esistenti;
  5. costruire il primo dataset reale;
  6. misurare copertura e casi difficili;
  7. nuove guide mirate **solo** per gli scenari mancanti.
- **Motivazione:** i qlog/rlog già presenti contengono GPS, encode-idx narrow/wide e `liveMapDataSP` → metadati di sincronizzazione delle guide originali. Le transizioni di `liveMapDataSP` servono a **suggerire clip candidate**, mai come ground truth.
- **Condizione:** ogni passo (accesso al comma, download) richiede autorizzazione esplicita in sessione; G1 resta **not established** finché il primo allineamento reale non è misurato.
- **Riesame:** in T2, prima di qualsiasi allineamento reale.

## D6 — Runner device e feasibility (APERTA, blocca G4)

- **Fatto:** la selezione stock vs tinygrad è param-driven (bundle attivo, `Runner` enum `snpe/tinygrad/stock`, default stock) — audit §2.3#1. Il runner installato sul comma non è determinabile dal Mac.
- **Scelta:** **nessuna**. G4 richiede misurazioni autorizzate (operator support, headroom, termica) sul runner reale. Nessun workload GPU è iniziato.
- **Riesame:** al primo benchmark autorizzato.

## D7 — Primo incremento: T1 APPROVATA (review 2026-09-18)

- **Decisione (approvata in review 2026-09-18):** avviare **T1** — pacchetto installabile, `speed_vision_core/types.py`, CLI `nnslr`, fixture malformed con reason code deterministici, test, quickstart CPU. Poi T2/T3 su fixture, poi **pausa** per esecuzione di D5 (video) e autorizzazione V100 per T4.
- **Vincoli T1 autorizzati:**
  ```text
  TRAIN_REPO: openpilot-nnslr-tools
  COMMITS:    local commits allowed
  PUSH:       forbidden
  COMMA:      forbidden
  V100:       forbidden
  GPU:        forbidden
  RUNTIME_REPO MODIFICATIONS: forbidden
  REAL DATASET REQUIRED: no
  ```
- **Exit criteria T1:** `pip install -e .` funziona; `nnslr --help` funziona; pytest completo passa; `types.py` applica le invarianti del §7; input malformed producono reason code deterministici; nessuna dipendenza da openpilot, GPU, dataset reale, device o modifiche a RUNTIME_REPO.
- **Non incluso (richiede autorizzazioni separate):** push su GitHub, download dal device, training su V100, qualsiasi modifica a `nn-speed-limit-vision`, qualsiasi lavoro sul comma.
- **Riesame:** alla fine di T1, stop per review prima di T2/T3 o di qualsiasi acquisizione hardware/dati.

## D8 — Architettura: VISION come fonte indipendente (registrata, review 2026-09-18)

- **Fatto (audit §2.3#2):** su Peugeot 3008, `carStateSP.speedLimit` ha consumer ma **nessun producer** nel runtime tree ispezionato.
- **Decisione:** il riconoscitore camera-based **non** deve mascherarsi da fonte CAR scrivendo in `carStateSP.speedLimit`. Le tre fonti restano concettualmente indipendenti:
  ```text
  MAP    → liveMapDataSP.speedLimit
  CAR    → carStateSP.speedLimit   (attualmente non prodotto su questa vettura)
  VISION → speedLimitVision        (nuova fonte indipendente camera-based)
  ```
- **Conseguenze:** nessun `@3` in `LongitudinalPlanSP.SpeedLimit.Source` per questi incrementi; nessun writer su `carStateSP`; la fusione/confronto tra fonti è compito di un eventuale resolver futuro che deve mantenere Vision identificabile come Vision (coerente con piano §4, §9 e vincolo permanente #1).
- **Riesame:** mai, per i incrementi A–D di questa specifica; una fonte operativa Vision è un progetto separato (piano §1).

## Vincoli permanenti (riassunto, non negoziabili in questa feature)

1. Vision **mai** nel control path: nessun input in `SpeedLimitResolver`, `speed_limit_assist.py`, `carStateSP.speedLimit`, enum `Source`, offset/confirmation, PSA CAN/radar/longitudinale/safety.
2. Default **disabled**; mai auto-attivazione per presenza di hardware/modello.
3. `unknown`/`unreadable`/`not_applicable`/`unavailable` restano esiti distinti; zero ≠ "senza limite".
4. Nessuna dipendenza dal server/Internet durante la guida; inference locale sul veicolo.
5. Dati privati (route, immagini, coordinate, checkpoint) **mai** in Git, mai nelle pipeline di upload esistenti (`uploader.py` comma + `sunnylink/uploader.py`).
6. Artifact promotion esplicita: bundle + core snapshot con digest, verificati; niente `latest`, niente auto-deploy.
7. Ogni blocco di codice modificato nei repo del port porta i marcatori `[tag] - START/END` (convenzione `openpilot_scripts/AGENTS.md`).
