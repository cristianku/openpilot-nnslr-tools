# NNSLR per Sunnypilot — Piano di implementazione e riapplicazione

> **Per gli agenti di sviluppo:** eseguire questo piano per incrementi verificabili, con test prima delle modifiche di comportamento. Usare `superpowers:executing-plans` o `superpowers:subagent-driven-development` quando disponibili; la loro assenza non impedisce di seguire manualmente i task. Le caselle indicano lavoro da svolgere, non autorizzazioni a pubblicare, utilizzare GPU o intervenire sulla vettura.

**Obiettivo:** integrare il riconoscimento dei cartelli di velocità in Sunnypilot come funzionalità indipendente e advisory, mantenendolo riapplicabile a una revisione aggiornata di `sunnypilot/sunnypilot:master` mediante una skill dedicata, un patchset Git versionato e verifiche riproducibili.

**Architettura:** due repository di codice. `openpilot-nnslr-tools` mantiene il core portabile, la pipeline di apprendimento e gli strumenti di riapplicazione; `cristianku/sunnypilot` mantiene gli adapter e gli hook runtime. Core e modelli sono artefatti fissati per hash; una nuova integrazione nasce da una nuova base upstream, non da una ricostruzione libera affidata alla memoria dell’agente.

**Tecnologie:** Python e Git per orchestrazione e verifiche; Cap’n Proto e messaging del fork per gli adapter; VisionIPC per i frame live; UI esistente di Sunnypilot. ONNX è l’artefatto di riferimento, non una garanzia di supporto sul dispositivo. Backend, Python, compilatore e dipendenze runtime seguono la revisione verificata del target.

**Specifica di riferimento:** `openpilot-nnslr-tools/docs/plan.md`, revisione v4, in particolare §§4.3, 7–13 e T9–T12. Questo documento ne dettaglia l’integrazione Sunnypilot e aggiunge il requisito di riapplicazione. Non ridefinisce i task T0–T12 né dichiara completati quelli del training. [P1]

**Documento:** revisione 1.0 · 18 settembre 2026 · piano, non implementazione o certificazione.

**Stato di questa consegna:** documento locale. Nessuna modifica ai due repository, nessuna installazione della skill, nessun accesso al Comma o alle V100. Tutti i nuovi percorsi, le API e i comandi di orchestrazione indicati sotto sono specifiche da implementare, salvo quando esplicitamente descritti come esistenti.

---

## Indice

1. [Baseline verificata e decisioni](#s01)
2. [Vincoli globali e autonomia dell’agente](#s02)
3. [Architettura, repository e file](#s03)
4. [Versioni, patchset e artefatti](#s04)
5. [Contratto runtime e stati](#s05)
6. [Backend, scheduling e Chestnut](#s06)
7. [Procedura di riapplicazione](#s07)
8. [Skill e interfaccia degli script](#s08)
9. [Piano esecutivo R00–R15](#s09)
10. [Matrice di test e non-interferenza](#s10)
11. [Gate di rilascio e prestazioni](#s11)
12. [Pubblicazione, installazione e rollback](#s12)
13. [Evidenze, rischi e decisioni ancora aperte](#s13)
14. [Definition of Done e ordine operativo](#s14)
15. [Fonti e tracciabilità](#s15)

<a id="s01"></a>
## 1. Baseline verificata e decisioni

### 1.1 Revisioni osservate

| Risorsa | Revisione verificata il 18 settembre 2026 | Interpretazione |
|---|---|---|
| `cristianku/openpilot-nnslr-tools:main` | `9ffa00057905025c9fe6c8226664acc52330c32d` | Sorgente del piano v4 e del tooling già pubblicato. Non prova il completamento della pipeline. |
| `cristianku/sunnypilot:nn-speed-limit-vision` | `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56` | Branch runtime iniziale del progetto. |
| `sunnypilot/sunnypilot:master` | `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56` | Upstream da cui derivare la prima integrazione. Coincide con la branch runtime al momento della verifica. |

Fonti: ref GitHub letti tramite il connettore. [B1][B2][B3]

**Non fissare questa base per sempre.** Ogni operazione “aggiorna al master corrente” deve recuperare nuovamente il ref upstream e congelare il suo SHA all’inizio dell’operazione. Una volta iniziati i test, non cambiare base a metà perché upstream avanza ancora.

### 1.2 Correzioni rispetto alle proposte precedenti

- Il runtime è **il repository `cristianku/sunnypilot`, branch `nn-speed-limit-vision`**. Non è un repository chiamato `cristianku/nn-speed-limit-vision`. I riferimenti storici del v4 a `cristianku/openpilot` e alla branch PSA non identificano il nuovo target. [B2][P1]
- Mantenere la nomenclatura del v4: package `openpilot/sunnypilot/speed_vision/`, daemon `speedvisiond`, messaggio `speedVisionState`. `speedLimitVisionSP` e `speed_limit_visiond` erano alternative discusse, non ulteriori servizi da implementare. [P1, §§7 e 12.5]
- Non inserire una futura fase di controllo automatico nella coda di questo piano. Il controllo basato su Vision resta un progetto separato. [P1, vincoli globali]
- Nessun secondo processo deve inizializzare Chestnut. Il sorgente ispezionato contiene un vincolo di proprietà del dispositivo in `modeld`; i due percorsi backend vanno trattati separatamente. [R1][P1, §10]
- Non usare il logging normale come scelta predefinita: il v4 prevede messaggi non registrati di default e diagnostica privata, facoltativa e limitata. [P1, §10.6]
- Non promettere che una patch testuale si applicherà senza adattamenti a qualsiasi futuro `master`. Il risultato richiesto è un **porting riproducibile, verificabile e recuperabile**.

### 1.3 Fatti, requisiti e proposte

Nel documento, un riferimento `[P1]` identifica un requisito del piano v4; `[R…]` identifica il sorgente runtime ispezionato; `[E…]` identifica documentazione esterna primaria. Le scelte operative introdotte qui — manifest della feature, strumenti, branch candidate, registri di compatibilità — sono **proposte progettuali di questo addendum**, non funzionalità già disponibili.

Non risultano verificati da questa attività: configurazione installata sul Comma, runner selezionato sul dispositivo, presenza o margine di Chestnut, modello NNSLR addestrato, correttezza end-to-end su route reali, prestazioni target e supporto delle regole stradali. Questi dati rimangono distinti dallo stato del codice GitHub.

<a id="s02"></a>
## 2. Vincoli globali e autonomia dell’agente

### 2.1 Invarianti da conservare

1. **Vision non produce controllo.** Nessuna modifica a CAN, radar, sicurezza, sterzo, ACC, coppie, accelerazioni, pulsanti cruise o pesi del driving model. Nessuna scrittura in `carStateSP.speedLimit` o `liveMapDataSP`.
2. **Default off.** Presenza del modello, connessione dell’acceleratore e aggiornamento di Sunnypilot non abilitano automaticamente NNSLR.
3. **SLA invariato rispetto alla nuova baseline.** Modalità, offset, conferme e politiche salvate dell’upstream scelto devono continuare a funzionare senza NNSLR come input.
4. **Core canonico unico.** Tracking, preprocessing di riferimento, applicabilità, passaggio e stato appartengono a `speed_vision_core` nel training repository. Nel runtime entra solo uno snapshot generato e verificato, non una seconda implementazione manuale.
5. **Inferenza locale.** Nessun server domestico, rete mobile o cloud richiesto durante la guida. Non installare il pacchetto training sul Comma.
6. **Incertezza esplicita.** `unknown`, `unreadable`, `not_applicable`, `unavailable` rimangono distinti. Nessuno equivale a zero o a velocità illimitata.
7. **Dati privati fuori Git.** Route, immagini, coordinate, checkpoint di training e report identificativi non vengono pubblicati automaticamente.
8. **Backend non verificato = disabilitato.** Non cambiare il driving-model bundle, i driver o le dipendenze globali del dispositivo per “far funzionare” NNSLR.

Questi vincoli derivano dal v4 e valgono anche dopo ogni riapplicazione. [P1, vincoli globali e §§4.3, 7–11]

### 2.2 L’agente non deve bloccarsi per ogni modifica

Dopo l’autorizzazione a sviluppare o riapplicare la feature, sono attività ordinarie: scrivere codice e test nei repository di sviluppo, correggere adapter, leggere sorgenti pubblici, elaborare file già locali, generare fixture e creare commit locali.

**“Non toccare il Comma” significa non accedere al dispositivo live.** Non vieta di sviluppare decoder, replay, adapter o test per i suoi formati.

L’agente chiede una decisione solo quando cambierebbe il perimetro: migrazione incompatibile del protocollo, uso di hardware non autorizzato, modifica di controlli, pubblicazione non autorizzata, aggiornamento distruttivo di un ref o installazione sulla vettura. Un import rinominato o un hook spostato, se adattabile senza cambiare il contratto, non richiede una nuova approvazione.

Questo piano da solo non avvia tali attività: definisce come svolgerle quando richieste.

<a id="s03"></a>
## 3. Architettura, repository e file

### 3.1 Separazione funzionale

```text
TRAIN_REPO: openpilot-nnslr-tools
  dataset → training → valutazione → bundle modello
  speed_vision_core → snapshot minimale + test golden
  skill + reapply tooling → nuova integrazione candidata
                       │
                       ▼
RUNTIME_REPO: cristianku/sunnypilot
  nuovo master upstream + patchset NNSLR
  adapter frame/context → backend → ObservationBatch
                                  │
                                  ▼
                            speedvisiond
                      core temporale / confronto
                                  │
                                  ▼
                          speedVisionState
                          │              │
                          ▼              ▼
                    UI separata    diagnostica privata

CAR / MAP → resolver SLA esistente → controlli esistenti
VISION    ─────────────── X ────────→ controlli esistenti
```

Il percorso Chestnut modifica solo la collocazione dell’inferenza: il runner già proprietario pubblica `speedVisionObservations`; `speedvisiond` resta l’unico publisher di `speedVisionState`. Non inserire detection nei campi di guida di `modelV2`. [P1, §§7.2 e 10]

### 3.2 File proposti nel training repository

```text
.github/skills/nnslr-reapply/SKILL.md
integration/sunnypilot/
  README.md
  feature.lock.json
  hook-map.json
  schema-bindings.json
  protected-surfaces.json
  compatibility/
    profiles.json
    migrations/
  docs/
    runtime-reapply-plan.md
    decisions.md
scripts/
  nnslr_reapply.py
src/nnslr_tools/integration/sunnypilot/
  __init__.py
  manifest.py
  git_ops.py
  discover.py
  apply.py
  verify.py
  reports.py
  cli.py
tests/integration/sunnypilot/
  test_manifest.py
  test_prepare.py
  test_apply.py
  test_conflicts.py
  test_idempotence.py
  test_reports.py
  test_skill_scenarios.py
```

`nnslr_reapply.py` è un wrapper sottile: non duplica la logica delle librerie. Non importa PyTorch e non inizializza accelerator hardware. La skill contiene istruzioni e collegamenti; non una seconda copia degli algoritmi di riapplicazione.

### 3.3 File proposti nel runtime

```text
openpilot/sunnypilot/speed_vision/
  __init__.py
  _vendor/speed_vision_core/
  artifacts.lock.json
  adapters.py
  runner.py
  publisher.py
  lifecycle.py
  diagnostics.py
  speedvisiond.py
  tests/
    test_core_snapshot.py
    test_schema_roundtrip.py
    test_frame_adapter.py
    test_runner_contract.py
    test_lifecycle.py
    test_control_isolation.py
    test_privacy.py
    test_ui_contract.py
    test_faults.py
    test_runtime_budget.py
openpilot/selfdrive/ui/sunnypilot/onroad/vision_speed_badge.py
```

I file saranno suddivisi ulteriormente solo quando una responsabilità concreta lo richiederà. Non creare cartelle vuote o backend fittizi per far sembrare completa l’architettura.

### 3.4 Punti di contatto con upstream

| Area | Percorso nella baseline | Cambiamento NNSLR ammesso |
|---|---|---|
| Schema custom | `openpilot/cereal/custom.capnp` | Struct dedicate; nessuna alterazione semantica delle struct operative. |
| Event union | `openpilot/cereal/log.capnp` | Binding verificato dei nuovi messaggi. |
| Servizi | `openpilot/cereal/services.py` | Registrazione esplicita, non-default-logged; limiti documentati. |
| Lifecycle | `openpilot/system/manager/process_config.py` | Avvio condizionale del solo processo opzionale. |
| Stato UI | `openpilot/selfdrive/ui/sunnypilot/ui_state.py` | Subscription opzionale e adattamento dello stato del badge. |
| Layout attivo | Rilevato nel nuovo audit | Inserimento di un widget separato senza riscrivere il renderer SLA. |
| Parametri | Registro effettivo rilevato nel nuovo audit | Chiavi NNSLR nuove, default off; nessuna riscrittura delle preferenze SLA. |
| Runner proprietario | `openpilot/selfdrive/modeld/modeld.py`, solo per il profilo supportato | Hook Chestnut opzionale solo dopo analisi e benchmark del percorso. |

La presenza dell’estensione UI e della selezione stock/tinygrad è verificata nel sorgente attuale. I percorsi e le firme vanno nuovamente risolti sul nuovo master. [R2][R4]

Non imporre un tetto artificiale di “sei file”: contano isolamento e minimalità delle modifiche, non un numero arbitrario. Un hook aggiuntivo necessario va dichiarato, motivato e testato.

### 3.5 Modelli e submodule NNLC

`openpilot/sunnypilot/neural_network_data` è attualmente un submodule del progetto `sunnypilot/neural-network-data`. [R5]

**Non è obbligatorio usarlo per NNSLR e non va modificato automaticamente.** Per la prima versione: bundle immutabile in una posizione locale autorizzata, selezionato attraverso `artifacts.lock.json`, con digest e formato verificati. L’eventuale distribuzione in release asset viene decisa separatamente. Nessun nuovo fork del submodule NNLC è un prerequisito di questo piano.

La posizione effettiva sul dispositivo verrà fissata dopo l’audit di aggiornamenti, upload, backup e condivisioni; non inventare un percorso “privato” senza verificarlo.

<a id="s04"></a>
## 4. Versioni, patchset e artefatti

### 4.1 Identità separate

Ogni candidato deve identificare almeno:

| Identità | Contenuto |
|---|---|
| Upstream base | Repository upstream, SHA completo, gitlink dei submoduli. |
| NNSLR runtime | Versione feature, commit sorgente del patchset, serie ordinata dei commit. |
| Core snapshot | Commit TRAIN_REPO, lista file ammessi, hash dei file e dell’albero canonico. |
| Modello | Digest del bundle, componenti detector/reader, contratto I/O, preprocessing, classi. |
| Regole e capability | Rule pack e profilo di capacità, con evidenza disponibile per ciascuna. |
| Ambiente | Lock e toolchain utilizzati per build/test; hardware/runner solo se misurati. |
| Porting | Base nuova, commit risultante, mapping old→new, report dei test e adattamenti. |

Un modello caricato senza errore non prova la parità numerica; un patchset applicato senza conflitti non prova la compatibilità; una build PC riuscita non abilita l’uso sul veicolo.

### 4.2 Fonte unica del patchset

Il codice runtime canonico resta nella branch/release NNSLR di `cristianku/sunnypilot`. `feature.lock.json` in TRAIN_REPO referenzia commit concreti di tale repository, senza duplicarne manualmente il codice.

La prima serie deve essere lineare e tematica. Gli aggiornamenti upstream non vengono mischiati ai commit della feature. Per ogni porting si conserva la serie precedente e si registra la nuova.

Formato richiesto per ogni voce della serie:

```text
logical_id        Identità stabile, es. nnslr.runtime.messaging
commit            SHA completo del commit sorgente
parent             SHA atteso del parent
required           true oppure profilo condizionale dichiarato
owned_paths        File NNSLR introdotti o modificati
integration_hooks  Hook upstream interessati
postconditions     Identificatori delle verifiche richieste
```

Non selezionare i commit cercando solo la parola `NNSLR` nel messaggio. Verificare parentela, ordine, contenuto e copertura della serie. Rifiutare merge commit non gestiti, commit estranei o un range che trascina vecchi aggiornamenti upstream.

Un export `git format-patch` può essere generato come artefatto di trasporto e backup, ma non diventa una seconda sorgente mantenuta a mano. `git range-diff` serve a mostrare le differenze fra serie, non a certificarne il comportamento. [E1][E2]

### 4.3 Contratto di `feature.lock.json`

Il validatore deve distinguere `bootstrap` da `sealed`:

| Campo | Regola |
|---|---|
| `schema_version` | Versione intera riconosciuta. |
| `feature_id` | `nnslr-speed-vision`. |
| `state` | `bootstrap` prima della prima release; `sealed` per una serie riapplicabile. |
| `runtime_repository` | `cristianku/sunnypilot`. |
| `upstream_repository`, `upstream_ref` | `sunnypilot/sunnypilot`, `refs/heads/master`. |
| `source_base_commit` | SHA completo della base del patchset. |
| `source_feature_commit` | SHA completo del risultato canonico; obbligatorio in stato `sealed`. |
| `patches` | Serie ordinata e verificata; non vuota per una feature `sealed`. |
| `core` | Commit e digest dello snapshot, con lista file ammessi. |
| `schema_bindings` | Identità dei messaggi e digest del loro contratto. |
| `capability_profile` | Riferimento al profilo, non autorizzazione implicita di hardware. |
| `hook_map_digest` | Hash della mappa degli hook sottoposta a review. |

Oggi la feature runtime non ha ancora una serie NNSLR da sigillare: la branch coincide con la base. Il bootstrap deve essere riportato come tale, **mai come riapplicazione riuscita di una feature vuota**. [B2][B3]

Evitare riferimenti circolari: un file committato nel runtime non può incorporare lo SHA del commit finale che lo contiene. Gli SHA finali appartengono al catalogo esterno in TRAIN_REPO e alla ricevuta di esecuzione; il lock runtime contiene core, modello e contratti.

### 4.4 Contratto stabile e migrazioni schema

Conservare type ID, ordinali, unità, flag di presenza e significato dei campi. La documentazione Cap’n Proto consente estensioni compatibili secondo regole precise; una rinominazione non autorizza una rinumerazione. [E3]

Alla prima integrazione verificare tutto `log.capnp` e `custom.capnp`, quindi registrare i binding scelti. La presenza attuale di struct riservate non garantisce che una di esse resterà inutilizzata in tutti i futuri fork. [R3]

Se upstream occupa un binding già usato da NNSLR:

- bloccare la promozione e produrre il dettaglio del conflitto;
- non scegliere automaticamente un altro slot;
- progettare una migrazione identificata per versione;
- conservare lo schema con cui leggere vecchi log/sidecar, senza decodificarli con il significato nuovo;
- provare producer, consumer e fixture vecchie/nuove secondo la matrice dichiarata.

Per i file condivisi controllare sottostrutture e campi protetti; non confrontare l’hash dell’intero `custom.capnp`, che NNSLR deve estendere legittimamente.

<a id="s05"></a>
## 5. Contratto runtime e stati

### 5.1 Input e output

Riutilizzare i tipi portabili del v4: `FrameRef`, `Detection`, `ObservationBatch`, `RoadContext`, `SignTrack`, `PerceptionHealth`, `LimitHypothesis`, `SourceSnapshot`, `AdvisoryComparison`. Implementazioni mancanti restano dipendenze esplicite del core nel training repository. [P1, §7.1]

`speedVisionState` deve contenere:

- identità schema, sessione/epoca del produttore, sequenza e hash degli artefatti;
- tempi separati di pubblicazione, ultimo frame elaborato, ultima osservazione e ultimo contesto;
- salute del backend, freschezza input, latenza, dropped frames, overflow e reason code;
- al massimo otto osservazioni/track per pubblicazione, con identità di frame, valore opzionale e prove;
- ipotesi advisory con stato, valore opzionale, intervallo di attivazione e motivo di invalidazione.

La proposta di epoca del produttore rende rilevabile anche un riavvio del solo daemon all’interno della stessa sessione del dispositivo. Deve invalidare lo stato ricordato, senza mescolare timestamp di domini differenti.

**Unità:** percezione/core `value_kph: int | None`; serializzazione dell’ipotesi `speedLimitMps` con flag di presenza. Per un dato di sola osservazione conservare `valueKph`. Non applicare gli offset SLA alle osservazioni NN. [P1, §§7.1–7.3]

### 5.2 Stati e limiti delle capacità

| Modalità | Comportamento |
|---|---|
| `off` | Nessuna inferenza NNSLR; nessun valore mostrato come valido. |
| `shadow` | Elaborazione autorizzata, nessun badge on-road; diagnostica privata solo se abilitata. |
| `observation` | “Camera: cartello osservato”; niente affermazione di limite corrente né avviso di superamento. |
| `advisory` | Ipotesi corrente solo con applicabilità, condizioni, passaggio e continuità validati. |

`VisionSpeedLimitMode`, `VisionSpeedLimitWarnings` e `VisionSpeedLimitDiagnostics` sono distinti dai parametri SLA esistenti. I warning restano off nella prima release visibile. [P1, §9]

Una capability dichiarata nel bundle non basta: deve coincidere con un profilo approvato per quella combinazione di core, modello, backend e target. Un toggle utente non può aggirare una capability non verificata.

### 5.3 Regole temporali essenziali

Un heartbeat nuovo non ringiovanisce un cartello vecchio. Tre elaborazioni dello stesso frame non valgono tre osservazioni. Due camere dello stesso istante non raddoppiano automaticamente l’evidenza. L’uscita del cartello dal campo visivo non dimostra il passaggio.

Per avere un’ipotesi corrente servono valore supportato, strada propria, condizioni risolte, passaggio e continuità. Un cartello letto bene su una rampa non diventa il limite dell’autostrada. Una cancellazione con semantica non risolta invalida l’ipotesi; non produce automaticamente 80, 120 o “nessun limite”. [P1, §§8–9]

Parametri iniziali ereditati dal v4, da congelare prima della valutazione: campionamento 5 Hz da confrontare con 10 Hz; consenso 3 osservazioni uniche su 5 entro 0,8 s; span minimo 0,2 s; freschezza massima 0,35 s. L’orizzonte di memoria di 300 s o 5 km è un limite ingegneristico di disponibilità, non una regola legale. [P1, §§8.1 e 8.5]

<a id="s06"></a>
## 6. Backend, scheduling e Chestnut

### 6.1 Percorso A — Backend indipendente

`speedvisiond` acquisisce il frame e usa un backend locale distinto dal proprietario di Chestnut. Un backend CPU permette test e benchmark iniziali, ma non va presentato come abbastanza veloce sul Comma prima di misurarlo.

L’interfaccia proposta in `runner.py` comprende `prepare`, `submit_latest`, `poll`, `health` e `close`. Gli input/output restano indipendenti dal framework. Il caricamento valida forme, dtype, canali, normalizzazione, classi, componenti e hash.

### 6.2 Percorso B — Inferenza nel proprietario Chestnut

Il runner già proprietario ospita il modulo ausiliario e pubblica `speedVisionObservations`. `speedvisiond` riceve i risultati e applica il core portabile.

Stock Python e native `modeld_tinygrad` sono integrazioni diverse: supportare una non dimostra l’altra. Il codice attuale seleziona il processo attraverso il bundle/runner attivo. Non cambiarlo per soddisfare NNSLR. [R1][R2]

Questo percorso richiede analisi di code, sincronizzazioni e fault condivisi. Un callback esistente, un thread o un `try/except` non sono prove di preemption GPU o isolamento. Un errore del driver condiviso può compromettere il processo principale; se non è gestibile entro i gate, il profilo rimane non disponibile. [P1, §10]

### 6.3 Regole comuni

Una sola elaborazione in-flight, un solo frame pending sostituibile; limiti espliciti a detection e crop. Nessuna attesa di NNSLR nel controllo primario. Nessun caricamento, download o compilazione first-use nel ciclo di guida.

Conservare buffer VisionIPC soltanto con una garanzia verificata di lifetime; altrimenti usare una copia posseduta e limitata. Preprocessing NV12 rispettoso di stride, piani, matrice colore e range. Non riutilizzare automaticamente l’immagine warped del driving model: potrebbe perdere cartelli laterali o sopraelevati. [P1, §§6.2 e 10.2]

Un modello mancante lascia la feature unavailable; non si sostituisce con un mock. I mock sono ammessi solo in test e replay esplicitamente sintetici, mai selezionabili come backend on-road.

<a id="s07"></a>
## 7. Procedura di riapplicazione

### 7.1 Workspace isolato

La procedura standard usa un **clone separato** per la candidata e uno per la baseline quando necessario. È un dettaglio operativo rispetto al generico “worktree isolato” del v4: la documentazione Git segnala ancora limitazioni dei linked worktree con submoduli. Un linked worktree è ammesso solo con un profilo verificato, non come requisito. [E4]

Mai operare nel checkout utilizzato per guidare. Mai richiedere di cancellare modifiche locali dell’utente per preparare una candidata: usare un’altra directory. Nessun `reset --hard`, `clean -fd`, `rebase` del ramo condiviso o aggiornamento forzato per iniziare.

### 7.2 Sequenza deterministica

| Passo | Azione | Verifica / output |
|---|---|---|
| 1 | Leggere lock e autorizzazioni dell’esecuzione. | Stato `sealed`, schema riconosciuto, repository e artefatti ammessi. Un lock `bootstrap` non entra in produzione. |
| 2 | Acquisire un lock esclusivo sul workspace di esecuzione. | Un secondo agente non può modificare la stessa candidata. |
| 3 | Recuperare il ref upstream richiesto. | Registrare SHA completo, URL effettivo e data di osservazione. In modalità offline usare uno SHA locale e non chiamarlo “master corrente”. |
| 4 | Recuperare la revisione sorgente della feature. | Verificare SHA, parentela e intera serie dichiarata; non fidarsi solo del nome del tag. |
| 5 | Preparare baseline e candidata sul nuovo SHA upstream. | Registrare build system, schema, runner, UI, registry parametri e gitlink. |
| 6 | Inizializzare i submoduli agli SHA registrati dalla nuova base. | Nessun `git submodule update --remote`. Eventuali variazioni del patchset vanno dichiarate e revisionate. |
| 7 | Eseguire la discovery degli hook. | Mappa univoca o errore di compatibilità; conservare i vecchi e nuovi percorsi. |
| 8 | Riapplicare i commit in ordine. | Un commit per responsabilità, con provenienza; nessuna risoluzione automatica `ours/theirs`. |
| 9 | Gestire differenze strutturali. | Adapter equivalenti + test aggiornati, oppure decisione bloccante. |
| 10 | Eseguire verifiche sorgente, protocollo, build e regressioni. | Risultato di ogni test: passato, fallito, non eseguito, non applicabile con motivazione. |
| 11 | Generare `range-diff`, diff verso la nuova base e ricevuta. | Tutte le modifiche estranee al patchset risultano spiegate o fanno fallire la verifica. |
| 12 | Sigillare localmente la candidata verificata. | Nuovo catalogo e branch identificabili; pubblicazione e deploy restano operazioni separate. |

`git submodule update --init --recursive --checkout` usa gli SHA del superproject; `--remote` cambia la sorgente delle revisioni selezionate e non appartiene a questa procedura. [E5]

### 7.3 Tre categorie di cambiamento upstream

**Meccanico:** import, nome o percorso spostato; firma adattabile senza cambiare semantica. L’agente corregge nel workspace candidato, aggiunge regression test e continua.

**Strutturale compatibile:** registry o lifecycle organizzato diversamente, nuovo layout UI, formato dei buffer modificato ma adattabile con prove. L’agente aggiorna `hook-map` e il profilo; non riusa automaticamente misure hardware precedenti. Se il contratto rimane invariato può proseguire con i test, dichiarando le verifiche da rifare.

**Architetturale o incompatibile:** binding schema occupato, ownership accelerator cambiata, impossibilità di mantenere lifetime/budget, alterazione dei controlli o significato nuovo di un campo. L’agente conserva tutto, descrive il conflitto e richiede una decisione prima della promozione. Può continuare altre attività indipendenti non bloccate.

Non saltare un commit diventato vuoto solo perché Git lo consente. Identificare se la funzionalità è stata realmente incorporata dall’upstream, duplicata o persa. Un commit “soddisfatto da upstream” ha una verifica comportamentale e un mapping esplicito nel report.

### 7.4 Idempotenza e riproducibilità

Con la stessa base, serie, configurazione e toolchain di trasformazione si deve ottenere lo stesso **albero sorgenti e insieme di gitlink**. Non pretendere gli stessi SHA dei nuovi commit: timestamp e metadati possono variarli. [Proposta di questo piano]

Una seconda esecuzione sullo stesso workspace con gli stessi input riconosce lo stato e non aggiunge servizi, chiavi, widget o commit duplicati. Input differenti richiedono una nuova candidata. Il manifest della run registra progressi sufficienti a riprendere dopo un’interruzione, ma non continua se lo stato locale è stato modificato fuori dalla procedura senza riconciliazione.

### 7.5 Non reintrodurre i vecchi submoduli

La nuova base può cambiare `msgq`, `tinygrad`, `opendbc` o altri gitlink. La riapplicazione non deve ripristinare gli SHA della precedente integrazione per comodità.

Confrontare esplicitamente i gitlink baseline/candidata. Il risultato deve coincidere con la nuova base, salvo variazioni NNSLR autorizzate e registrate. Un vecchio binario compilato per un altro runtime non supera il gate perché il relativo file ha ancora lo stesso nome.

### 7.6 Branch stabile e history non fast-forward

Nome iniziale: `nn-speed-limit-vision`. Nuove candidate: `nnslr/reapply/<base-abbreviata>-<feature-version>-<run-id>`.

Ricostruire la feature sopra una nuova base può produrre una storia **non fast-forward** rispetto alla vecchia branch NNSLR. Questo è previsto, non un errore da “risolvere” con un force-push automatico.

Scelta predefinita: pubblicare, quando autorizzato, una nuova branch/release e conservare quella precedente. Un eventuale spostamento dell’alias stabile richiede una decisione esplicita, verifica degli updater che lo seguono, conservazione del vecchio SHA e protezione contro aggiornamenti concorrenti. Non sostituire il requisito di storia pulita con un merge della vecchia integrazione senza dichiararlo.

<a id="s08"></a>
## 8. Skill e interfaccia degli script

### 8.1 Ruolo della skill

Nome: **`nnslr-reapply`**. Percorso canonico: `.github/skills/nnslr-reapply/SKILL.md` in TRAIN_REPO.

La documentazione Copilot descrive skill come directory con `SKILL.md`, frontmatter e risorse opzionali. Il caricamento va verificato nel client utilizzato: il nome di una slash command non è una prova che la skill sia stata installata. Una copia personale deve derivare da una revisione fissata, non essere mantenuta manualmente in parallelo. [E6][E7]

Per usarla dal progetto runtime, aprire il workspace che include TRAIN_REPO oppure installare la stessa skill in una posizione personale supportata dal client. Se il client non la carica, l’agente può leggere esplicitamente il documento e usare gli script. Non dichiarare che una skill è stata eseguita senza averne caricato istruzioni e risorse.

### 8.2 Testo proposto per `SKILL.md`

Il seguente contenuto è una specifica da implementare e collaudare, non una skill installata da questa consegna.

```markdown
---
name: nnslr-reapply
description: Usare quando si deve ricostruire NNSLR Speed Limit Recognition sopra un nuovo master di Sunnypilot o verificare la compatibilità di un patchset NNSLR esistente.
---

# Riapplicazione NNSLR a Sunnypilot

Leggi il piano runtime-reapply e il feature.lock.json della revisione
TRAIN_REPO selezionata. Verifica se il lock è bootstrap oppure sealed.
Non ricostruire il codice dalla cronologia della chat.

Usa scripts/nnslr_reapply.py per inspect, prepare, apply, verify e report.
Recupera sunnypilot/sunnypilot:master una volta e registra lo SHA.
Lavora in un clone candidato isolato; conserva base e release precedenti.

Riapplica la serie registrata. Per import, percorsi e hook equivalenti,
correggi gli adapter, aggiungi test e continua senza chiedere permesso
per ogni riga. Non risolvere conflitti scegliendo indiscriminatamente
ours/theirs e non eliminare test per rendere verde la suite.

Conserva il core portabile canonico, il protocollo versionato, default off,
assenza di input Vision ai controlli, timestamp originali e log privati.
Non inizializzare Chestnut da un secondo processo.

Non accedere al Comma o alle GPU, non pubblicare e non installare senza
l'autorizzazione pertinente. Questo non vieta sviluppo, replay e test
su file locali. Non introdurre interruzioni per tale falso motivo.

Produci diff dalla nuova base, range-diff fra le serie, test eseguiti,
test non eseguiti, capacità confermate/non confermate e prossima azione.
Un'applicazione pulita non equivale a un rilascio target validato.
```

### 8.3 CLI da implementare

Questi comandi sono **l’interfaccia prevista**, non comandi già presenti in `nnslr`:

```bash
# TRAIN_REPO: validazione dei metadati senza fetch né modifiche al runtime.
python scripts/nnslr_reapply.py inspect \
  --feature-lock integration/sunnypilot/feature.lock.json

# Crea un ambiente isolato e fissa lo SHA corrente upstream.
python scripts/nnslr_reapply.py prepare \
  --feature-lock integration/sunnypilot/feature.lock.json \
  --upstream-ref refs/heads/master \
  --workspace /path/to/nnslr-porting

# La prepare restituisce il percorso della run da usare nei comandi seguenti.
python scripts/nnslr_reapply.py apply --run-dir "$NNSLR_PORT_RUN"
python scripts/nnslr_reapply.py verify --run-dir "$NNSLR_PORT_RUN" --profile source
python scripts/nnslr_reapply.py report --run-dir "$NNSLR_PORT_RUN"

# Solo dopo esiti richiesti disponibili; non pubblica né installa.
python scripts/nnslr_reapply.py seal --run-dir "$NNSLR_PORT_RUN"
```

Per riproduzione storica, `prepare --upstream-sha` sostituisce `--upstream-ref`; le due opzioni si escludono. La variabile `NNSLR_PORT_RUN` va valorizzata dall’output reale di `prepare`, non dal nome inventato di una run.

### 8.4 Contratto API dell’orchestratore

| Modulo | Interfaccia proposta | Effetto |
|---|---|---|
| `manifest.py` | `load_feature_lock(path) -> FeatureLock` | Validazione strict e risoluzione degli input. |
| `discover.py` | `discover_hooks(repo, profile) -> HookMap` | Identifica gli hook senza modificarli. |
| `git_ops.py` | `prepare_run(request) -> PreparedRun` | Clone, fetch, SHA congelati e ricevuta iniziale. |
| `apply.py` | `apply_patchset(run) -> ApplyResult` | Applicazione ordinata; conflitti strutturati. |
| `verify.py` | `verify_run(run, profile) -> VerificationReport` | Esegue soltanto le prove richieste dal profilo. |
| `reports.py` | `write_report(run, result) -> ReportPaths` | JSON e Markdown con dati sanitizzati. |
| `manifest.py` | `seal_candidate(run, verification) -> SealedCandidate` | Sigilla solo se i requisiti del livello dichiarato sono soddisfatti. |

`PreparedRun` contiene input congelati, directory, candidata e baseline. `ApplyResult` contiene mapping dei commit, adattamenti e conflitti. `VerificationReport` contiene risultati nominati e livello raggiunto; non soltanto `success: true`.

Codici di uscita: `0` operazione richiesta riuscita; `2` input/ambiente non valido; `3` conflitto di applicazione; `4` contratto incompatibile; `5` test fallito; `6` verifica richiesta non stabilita. L’esito `0` di `prepare` non afferma che build o runtime siano stati validati.

Timeout, directory preesistente, lock concorrente, oggetto Git mancante, submodule non accessibile e output non scrivibile devono produrre un errore nominato, senza cancellare input o lavoro precedente.

<a id="s09"></a>
## 9. Piano esecutivo R00–R15

Gli identificativi **R** appartengono a questo addendum. Le dipendenze **T/G** restano quelle del v4. Ogni task di codice termina con: test mirato inizialmente rosso, implementazione, test verde, regressioni vicine, documentazione e commit locale nominativo. Non usare il conteggio dei commit o delle righe come percentuale di completamento.

### R00 — Baseline, mappa del target e decisioni

**Repository:** entrambi, lettura dei sorgenti; output documentale in TRAIN_REPO.

**File:** `integration/sunnypilot/docs/decisions.md`, `hook-map.json`, `protected-surfaces.json`.

**Input/output:** ref reali e v4 → baseline fissata e distinzione fra hook, superfici protette e capacità sconosciute.

- [ ] Registrare SHA upstream/fork/training, gitlink, build system e struttura dei package.
- [ ] Correggere nel piano operativo i vecchi nomi del runtime senza alterare la provenienza del v4 storico.
- [ ] Tracciare produzione e consumo di Car/Map/SLA, parametri, layout UI, logging e uploader.
- [ ] Verificare la mappa degli hook con test di discovery su sorgente fissato.
- [ ] Registrare separatamente le informazioni non disponibili sul dispositivo; non inventare runner o GPU.

**Accettazione:** nessun hook viene indicato come presente sulla sola base di un vecchio numero di riga. Test `test_discovery_matches_pinned_baseline`; commit `docs(nnslr): record runtime baseline and integration boundaries`.

### R01 — Manifest della feature e supporto bootstrap

**Repository:** TRAIN_REPO.

**File:** `src/nnslr_tools/integration/sunnypilot/manifest.py`, `git_ops.py`; test `test_manifest.py`, `test_prepare.py`; lock e schema bindings.

**Interfaccia:** `FeatureLock`, `PreparedRun`, `load_feature_lock`, `prepare_run` secondo §8.4.

- [ ] Scrivere fixture di repository Git minuscoli locali: base, due commit feature, upstream successivo.
- [ ] Testare lock incompleto, SHA errato, serie non lineare, parent incoerente e commit estraneo.
- [ ] Implementare `bootstrap` senza dichiarare una serie vuota riapplicabile.
- [ ] Implementare il clone separato e l’esclusione fra `--upstream-ref` e `--upstream-sha`.
- [ ] Testare directory già occupata e due esecuzioni concorrenti; nessuna sovrascrittura.

**Accettazione:** test offline della procedura, senza repository privati né dispositivo. `python -m pytest tests/integration/sunnypilot/test_manifest.py tests/integration/sunnypilot/test_prepare.py -q`.

### R02 — Snapshot del core e handoff degli artefatti

**Repository:** TRAIN_REPO canonico; RUNTIME_REPO consumatore.

**File:** exporter `src/nnslr_tools/bundle.py` o modulo dedicato se già organizzato diversamente; `_vendor/speed_vision_core/`, `artifacts.lock.json`; `test_core_snapshot.py`.

**Input/output:** commit core selezionato e lista ammessa → snapshot minimale con licenza, hash e fixture golden.

- [ ] Testare rigetto di file training, import torch, file privati e modifiche non registrate nel sorgente selezionato.
- [ ] Generare lo snapshot, senza editing manuale dei file vendored.
- [ ] Provare import senza framework training e senza adattatori live.
- [ ] Confrontare output reference/vendored su stessi frame sintetici, timestamp e stati.
- [ ] Verificare contratto e hash di ogni componente modello; modello assente significa capability assente.

**Accettazione:** parità del core e rigetto di artefatti incompatibili. Non è richiesta una rete già addestrata per verificare il formato, ma non si dichiara inferenza reale.

### R03 — Schema, codec e servizi dedicati

**Repository:** RUNTIME_REPO.

**File:** `custom.capnp`, `log.capnp`, `services.py`, `speed_vision/publisher.py`, `test_schema_roundtrip.py`.

**Input/output:** tipi core → `speedVisionState`; `speedVisionObservations` solo per il profilo co-locato.

- [ ] Audit completo dei binding e registrazione stabile di quelli scelti.
- [ ] Testare presenza/assenza del valore, unità, timestamp, schema sconosciuto e liste oltre il limite.
- [ ] Implementare codec con campi espliciti; nessun target di velocità o richiesta di attuazione.
- [ ] Registrare i servizi non-default-logged e testarne effettivamente il comportamento.
- [ ] Compilare schema/header con la toolchain del target e verificare fixture di compatibilità.

**Accettazione:** un solo publisher per servizio; `hasValue=false` non genera un limite zero; enum e campi operativi preesistenti invariati rispetto alla baseline. Commit `feat(nnslr): add isolated advisory message contract`.

### R04 — Adapter camera e contesto con lifetime verificato

**Repository:** RUNTIME_REPO.

**File:** `speed_vision/adapters.py`, `test_frame_adapter.py`, fixture camera senza dati privati.

**Input/output:** VisionIPC e messaggi read-only → `FrameRef`, immagine posseduta e `RoadContext` senza truth labels.

- [ ] Testare NV12 con stride/padding, dimensioni differenti e metadati mancanti.
- [ ] Testare buffer riutilizzato dal produttore: l’elaborazione non deve leggere pixel cambiati.
- [ ] Conservare SOF/EOF, stream, frame ID e sessione; nessuna associazione per “ultimo messaggio disponibile”.
- [ ] Restituire contesto unknown se il frame di contesto non è correlabile entro il budget dichiarato.
- [ ] Confrontare preprocessing di riferimento e adapter su golden input e coordinate native.

**Accettazione:** nessun frame vecchio associato al contesto nuovo; memoria limitata; test specifico `test_reused_vipc_buffer_does_not_change_owned_snapshot`.

### R05 — Backend indipendente e contratto del runner

**Repository:** RUNTIME_REPO; fixture export/reference in TRAIN_REPO.

**File:** `speed_vision/runner.py`, `test_runner_contract.py`; profilo compatibilità.

**Input/output:** immagine e artefatti verificati → `ObservationBatch` oppure indisponibilità con reason code.

- [ ] Definire e testare `prepare`, `submit_latest`, `poll`, `health`, `close`.
- [ ] Provare modello mancante, reader mancante, forme errate, valori non finiti e risultato fuori tempo.
- [ ] Implementare un percorso di riferimento utilizzabile offline; backend target solo per profili verificati.
- [ ] Verificare ogni operatore e preprocessing rispetto al runtime fissato, non a quello installato sul server training.
- [ ] Testare che nessun fallback abiliti un mock o selezioni un altro driving model.

**Accettazione:** compatibilità dichiarata per combinazione esatta; CPU reference non equivale a prontezza on-device. L’effettivo carico su hardware richiede l’autorizzazione del relativo benchmark.

### R06 — Daemon, lifecycle, salute e code limitate

**Repository:** RUNTIME_REPO.

**File:** `speed_vision/speedvisiond.py`, `lifecycle.py`, `process_config.py`, registry parametri individuato in R00; `test_lifecycle.py`, `test_faults.py`.

- [ ] Testare default off, avvio senza modello, camera assente, restart e cambio modalità.
- [ ] Limitare in-flight/pending; sostituire frame pending obsoleti invece di accodarli.
- [ ] Separare heartbeat, ultimo capture elaborato e ultima evidenza; invalidare su nuova epoca del producer.
- [ ] Implementare fault latched del backend e arresto del solo lavoro ausiliario.
- [ ] Verificare che l’assenza del processo opzionale non renda obbligatorio il suo heartbeat per l’engagement primario.

**Accettazione:** disabilitare NNSLR rimuove lavoro e usabilità advisory senza modificare i parametri SLA. Nessun loop di restart non limitato che consumi risorse del sistema principale.

### R07 — Confronto advisory e capacità interpretative

**Repository:** logica in TRAIN_REPO `speed_vision_core/advisory.py` e moduli core; adapter runtime soltanto.

**Dipendenze:** contratti R02–R06; per l’ipotesi corrente, task interpretativi del v4 e gate G3.

- [ ] Testare osservazione con ownership/passaggio unknown: non deve diventare corrente.
- [ ] Testare Car/Map assenti, conflitto, età ignota e tempi in domini diversi.
- [ ] Conservare snapshot originali senza cast silenziosi né moltiplicazione della confidence per accordo mappa-camera.
- [ ] Pubblicare osservazione quando le capability interpretative non sono validate.
- [ ] Implementare o completare tracking, ownership, passaggio e continuità nel core canonico soltanto; esportare poi un nuovo snapshot.

**Accettazione:** modalità `advisory` non aggira G3; assenza di training/interpretazione reale è esplicita. Nessun riempimento del valore dal limite mappa.

### R08 — Widget UI separato

**Repository:** RUNTIME_REPO.

**File:** `vision_speed_badge.py`, estensione UI state e layout attivo; `test_ui_contract.py`.

- [ ] Scrivere fixture per off, shadow, observed, ahead, current, stale, conflict e unavailable.
- [ ] Testare che shadow non mostri il badge e che observed/ahead non generino warning di superamento.
- [ ] Aggiungere il widget senza sostituire il badge SLA operativo.
- [ ] Verificare metrica/imperiale, dimensioni display, priorità degli alert e modalità Assist attiva.
- [ ] Rendere gli score dettagliati visibili solo in replay/debug, non come rumore permanente in guida.

**Accettazione:** UI testabile offline prima del modello; abilitazione sul veicolo solo al gate previsto. Un valore storico può essere mostrato come storico, mai come corrente.

### R09 — Diagnostica privata e isolamento degli upload

**Repository:** RUNTIME_REPO.

**File:** `speed_vision/diagnostics.py`, `test_privacy.py`; inventario delle esposizioni in `integration/sunnypilot/docs/decisions.md`.

- [ ] Tracciare logger, uploader standard, sunnylink, backup, crash log e condivisioni HTTP/file.
- [ ] Testare i messaggi NNSLR contro il routing del logger, non solo contro `services.py`.
- [ ] Scegliere una directory soltanto dopo averne verificato esclusioni e permessi.
- [ ] Applicare rotazione limitata: proposta v4, dieci file da massimo 32 MiB; nessun raw frame per default.
- [ ] Simulare disco pieno/lento, coda diagnostica piena ed eccezioni; scartare diagnostica senza bloccare il lavoro primario.

**Accettazione:** nessun payload privato nei canali automatici. Il report sorgente pubblico contiene solo identificativi sanitizzati e digest; le route reali rimangono private.

### R10 — Regressioni, controlli protetti e CI

**Repository:** entrambi, secondo l’ownership dei test.

**File:** `test_control_isolation.py`, verificatore di `protected-surfaces.json`, workflow CPU/source e profili test.

- [ ] Eseguire i test baseline prima delle modifiche per distinguere regressioni da difetti preesistenti.
- [ ] Iniettare Vision assente, valida, conflittuale, stale, malformed e faulted sugli stessi input registrati.
- [ ] Confrontare target, attuatori, source operative, set-speed e transizioni della baseline/candidata.
- [ ] Verificare che il patchset non modifichi file protetti; nei file condivisi confrontare le strutture operative.
- [ ] Separare test pure CPU, runtime build, replay privato e benchmark hardware. Registrare marker `hardware` e `private_data`; nessun accesso GPU/device durante collection o import.

**Accettazione:** nessun test essenziale saltato entra nel conteggio “passato”. Niente CI non attendibile sul server domestico con credenziali/dataset; niente deploy automatico. [P1, §§11.3 e 12.6]

### R11 — Adapter Chestnut, solo se scelto e fattibile

**Repository:** RUNTIME_REPO.

**File:** hook del runner verificato, `speed_vision/runner.py`, codec interno; `test_runtime_budget.py`, `test_faults.py`.

- [ ] Modellare con test l’admission control: budget insufficiente deve scartare il lavoro opzionale.
- [ ] Stabilire l’esatto punto di integrazione nel runner proprietario e l’ownership di code/buffer.
- [ ] Preparare modello e allocazioni prima del punto di esecuzione validato.
- [ ] Pubblicare sul servizio interno senza bloccare il runner in attesa del consumer.
- [ ] Eseguire benchmark autorizzati concorrenti e fault al banco; non simulare recovery resettando la GPU in guida.

**Accettazione:** G4/G5 per quella combinazione; in caso negativo il percorso rimane disabilitato. R11 è opzionale: un backend indipendente accettabile non obbliga ad aggiungere Chestnut.

### R12 — Motore di riapplicazione completo

**Repository:** TRAIN_REPO.

**File:** `apply.py`, `discover.py`, `verify.py`, `reports.py`, `scripts/nnslr_reapply.py`; test `test_apply.py`, `test_conflicts.py`, `test_idempotence.py`, `test_reports.py`.

- [ ] Prima testare serie su stessa base, base avanzata, hook spostato, binding occupato e commit già assorbito.
- [ ] Implementare applicazione ordinata con mapping di provenienza e persistenza dello stato.
- [ ] Verificare zero cambiamenti esterni al workspace candidato e nessun gitlink ripristinato per errore.
- [ ] Generare diff verso nuova baseline, `range-diff` e risultati strutturati.
- [ ] Testare resume dopo interruzione e seconda esecuzione idempotente.

**Accettazione:** `verify --profile source` dimostra soltanto il livello source; non promuove automaticamente il candidato a device-ready.

### R13 — Skill collaudata come procedura

**Repository:** TRAIN_REPO.

**File:** `.github/skills/nnslr-reapply/SKILL.md`, README integrazione, `test_skill_scenarios.py` e protocollo di review.

- [ ] Eseguire scenari senza la skill per registrare gli errori procedurali che deve prevenire.
- [ ] Integrare le istruzioni brevi di §8 con link agli script, senza duplicarne i controlli.
- [ ] Ripetere gli scenari con la skill caricata e confrontare le azioni realmente compiute.
- [ ] Verificare in particolare: non fermarsi sul trattamento di file locali; non forzare un ref; non rinumerare schema; non ignorare test mancanti.
- [ ] Testare l’uso dal workspace corretto e l’eventuale copia personale versionata.

**Accettazione:** risultato documentato nel client effettivamente usato. Se non è disponibile un harness multi-agente, la verifica può essere manuale e ripetibile; non va dichiarata automatica.

### R14 — Replay, bench e shadow secondo i gate originali

**Repository:** replay/evaluation in TRAIN_REPO; prove specifiche target in RUNTIME_REPO.

**Dipendenze:** modello utile e compatibile, dataset verificato, profilo backend definito, autorizzazione all’hardware per le prove live.

- [ ] Verificare route e frame/log alignment prima di misurare latenza di passaggio.
- [ ] Eseguire replay senza etichette oracle come input al runtime; i dati annotati servono al confronto.
- [ ] Eseguire A/B baseline/candidata, cold/warm start, scene cariche, restart e fault.
- [ ] Registrare metriche primarie, latenza end-to-end, temperatura, memoria e dropped frames.
- [ ] Solo dopo G5, effettuare shadow autorizzato; solo dopo G6 abilitare observation UI, e current advisory soltanto con G3.

**Accettazione:** non confondere schema funzionante, test sintetico, benchmark hardware e precisione su strada. Misure insufficienti significano gate non stabilito.

### R15 — Seconda base, release riproducibile e rollback

**Repository:** entrambi; nessuna pubblicazione implicita.

- [ ] Sigillare il primo patchset reale con SHA e snapshot verificati.
- [ ] Ricostruirlo sulla stessa base e confrontare gli alberi.
- [ ] Riapplicarlo su una seconda revisione upstream realmente distinta e disponibile; prima che esista, usare una fixture modificata senza dichiararla prova reale.
- [ ] Verificare compatibilità, test richiesti e nuovi benchmark quando invalidati dal cambiamento.
- [ ] Preparare release note, profilo supportato, vecchia release conservata e istruzioni di rollback off-road.
- [ ] Pubblicare candidata/tag soltanto se autorizzato, senza modificare la branch installata implicitamente.

**Accettazione:** esiste evidenza concreta della riapplicazione e della conservazione della vecchia versione. Nessuna promessa di compatibilità universale futura.

<a id="s10"></a>
## 10. Matrice di test e non-interferenza

### 10.1 Casi minimi obbligatori

| ID | Scenario | Risultato richiesto |
|---|---|---|
| RP01 | Stessa base/serie eseguita due volte | Stesso tree; nessun hook/commit duplicato. |
| RP02 | Upstream sposta un registry o import | Adapter aggiornato, test e mapping; nessuna ricostruzione globale del progetto. |
| RP03 | Upstream usa uno slot schema NNSLR | Incompatibilità esplicita; nessuna rinumerazione silenziosa. |
| RP04 | Nuovi gitlink upstream | Conservati; non sostituiti dai vecchi gitlink della feature. |
| RP05 | Worktree utente sporco | Rimane intatto; candidata in clone separato. |
| RP06 | Remote cambia dopo il fetch iniziale | Run resta sulla base congelata; eventuale nuova richiesta crea un’altra run. |
| RP07 | Commit feature già integrato upstream | Verifica di equivalenza e mapping; non duplicazione o skip cieco. |
| RT01 | NNSLR off | Nessun carico ausiliario previsto e nessuna modifica semantica ai controlli. |
| RT02 | Valore NN corretto ma other-road | Nessuna ipotesi corrente per la strada propria. |
| RT03 | Tre osservazioni dello stesso frame | Nessun consenso temporale artificiale. |
| RT04 | Nuova pubblicazione con vecchia detection | Età dell’evidenza invariata; scadenza rispettata. |
| RT05 | Backend cambia sessione/epoca | Stato precedente invalidato. |
| RT06 | Panel illeggibile o non interamente visibile | Condizione unresolved, anche se il numero è leggibile. |
| RT07 | Cartello scompare dietro un camion | Nessuna deduzione automatica di passaggio. |
| RT08 | Disaccordo MAP/CAR/VISION | Conflitto advisory; resolver operativo invariato. |
| RT09 | Decoder/modello produce NaN o overflow | Risultato rifiutato, motivo stabile, lavoro limitato. |
| RT10 | Consumer UI lento o diagnostica piena | Nessuna attesa introdotta nel processo primario. |
| RT11 | Cambio del driving runner/bundle | Capability target da riverificare; nessun cambio automatico del driving model. |
| RT12 | Camera/buffer cambiano dimensioni o formato | Rigetto o profilo verificato; niente reshape arbitrario. |
| PR01 | Logger/uploader/backup/file sharing attivi | Nessuna diffusione dei payload NNSLR privati. |
| AR01 | Snapshot core o reader mancante/alterato | Bundle rifiutato; niente sostituzione implicita. |

### 10.2 Assert di riferimento per implementatori

Gli esempi descrivono test da creare con le fixture previste dai task; non sono risultati già eseguiti.

```python
# RP01: confrontare il contenuto, non l'identità dei nuovi commit.
assert first_run.result_tree == second_run.result_tree
assert first_run.result_gitlinks == second_run.result_gitlinks
assert second_apply.added_hook_count == 0

# Non-interferenza: baseline = il nuovo upstream selezionato.
assert candidate_outputs.actuator_requests == baseline_outputs.actuator_requests
assert candidate_outputs.operational_sla == baseline_outputs.operational_sla
assert candidate_outputs.cruise_requests == baseline_outputs.cruise_requests

# RT04: la pubblicazione nuova non modifica l'origine dell'evidenza.
assert republished.lastObservationMonoNs == original.lastObservationMonoNs
assert not stale_hypothesis.usableForAdvisory

# RP03: nessun successo parziale mascherato.
assert schema_conflict_report.compatible is False
assert schema_conflict_report.promotion_allowed is False
```

Le proiezioni `actuator_requests`, `operational_sla` e `cruise_requests` vanno definite dal test harness sul protocollo della baseline. Ogni campo escluso dal confronto deve essere elencato e motivato come non semantico; non ignorare campi per eliminare una regressione.

### 10.3 Test da eseguire per livello

```bash
# TRAIN_REPO — nomi dei nuovi test definiti in questo piano.
python -m pytest tests/integration/sunnypilot -q
python -m pytest tests/core tests/cli -q

# RUNTIME_REPO — dopo preparazione della toolchain e dei submoduli fissati.
python -m pytest openpilot/sunnypilot/speed_vision/tests -m "not hardware and not private_data" -q
python -m pytest openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/tests -q
```

I marker `hardware` e `private_data` sono da registrare in R10: il profilo source seleziona esplicitamente i test senza tali requisiti. Un test source obbligatorio saltato per un errore di ambiente lascia il gate non stabilito. I profili hardware e replay privato vanno invocati separatamente dopo autorizzazione.

Il target di build, i test del manager e quelli UI vanno identificati in R00 e salvati nel profilo di compatibilità; non inventare un comando universale di build valido per qualunque futuro layout.

Una ricerca statica dei consumer è necessaria ma non sufficiente. Un test di import o una suite con tutti i casi hardware saltati non prova che il daemon funzioni sul dispositivo.

<a id="s11"></a>
## 11. Gate di rilascio e prestazioni

### 11.1 Non creare gate paralleli incompatibili con il v4

| Gate v4 | Evidenza necessaria | Conseguenza runtime |
|---|---|---|
| G0 | Baseline e ownership accettate | Sviluppo locale nel perimetro richiesto. |
| G1 | Dati, sincronizzazione, annotazioni e split attendibili | Confronti di modelli significativi. |
| G2 | Percezione utile, parity export e metriche indipendenti | Dimostrazione observation-only offline. |
| G3 | Ownership, condizioni, passaggio e continuità non-oracle | Capacità di ipotesi corrente, non abilitazione dei controlli. |
| G4 | Fattibilità backend sul target autorizzato | Integrazione al banco. |
| G5 | Non-interferenza semantica e runtime, fault, disable/rollback | Possibilità di shadow, con autorizzazione separata. |
| G6 | Shadow rappresentativo e review dei problemi | Observation UI; corrente solo con G3. |
| G7 | UI, support matrix e capacità effettive approvate | Release advisory nel dominio dichiarato. |

Fonte e significato: v4 §13. Una nuova build non eredita automaticamente tutti i gate della precedente. [P1]

### 11.2 Livelli di risultato della riapplicazione

La ricevuta distingue almeno:

- **`source_ready`**: patchset, contratti, test CPU e build richiesti dal profilo superati.
- **`bench_ready`**: oltre al sorgente, backend e benchmark autorizzati soddisfano G4/G5.
- **`shadow_accepted`**: evidenza G6 valida per la combinazione esatta.
- **`advisory_released`**: G7 soddisfatto, con le sole capability validate.

Uno stato non sostituisce l’autorizzazione all’operazione successiva. In particolare, `source_ready` non significa “installa adesso sulla Peugeot”.

### 11.3 Budget iniziali del v4, non misure già ottenute

| Metrica | Target iniziale da congelare prima della valutazione |
|---|---|
| Nuovi deadline miss attribuibili a NNSLR | Zero nelle prove A/B dichiarate; gli errori baseline restano separati. |
| Regressione p99 del runner principale | Non oltre il minore fra 1 ms e 2% della p99 baseline, su prove ripetute. |
| Capture-to-result | p95 ≤150 ms, p99 ≤200 ms al campionamento selezionato. |
| Freschezza per un nuovo evento | Risultati oltre 350 ms non creano nuovi eventi advisory. |
| Coda pending | Massimo uno; frame obsoleti scartati. |
| Blocco ausiliario nel driving loop | Nessuna attesa di output, log o consumer NNSLR. |
| Memoria e termica | Stabilità dopo warm-up; headroom e throttling misurati. |

Protocollo iniziale: almeno tre prove baseline e tre candidate, con dieci minuti di warm-up e trenta di regime per prova, ordine alternato e condizioni registrate. Sono proposte ingegneristiche del v4, non certificazioni né prova di worst-case execution time. [P1, §10.4]

### 11.4 Quando una nuova base invalida le prove precedenti

**Proposta operativa:** registrare la motivazione dell’eredità o dell’invalidazione di ogni prova.

| Cambiamento | Verifiche minime da ripetere |
|---|---|
| Solo documentazione | Integrità della serie e checks sorgente applicabili. |
| Schema/messaging/registry | Codec, compatibilità, publisher/consumer, routing log e regressioni. |
| UI o unità di misura | Snapshot UI, stato, warning, metrica/imperiale e SLA separato. |
| VisionIPC, preprocessing o timestamp | Golden input, lifetime, sincronizzazione, replay e misure end-to-end. |
| Runner, accelerator, driver, toolchain o submodule di esecuzione | Operatori, parity, G4/G5, fault e benchmark target. |
| Modello, classi o core interpretativo | Replay, capability interessate e valutazione indipendente; gate target se cambia carico/comportamento. |

La skill può identificare quali prove siano necessarie; non può sostituire misurazioni mancanti con un giudizio testuale di plausibilità.

<a id="s12"></a>
## 12. Pubblicazione, installazione e rollback

### 12.1 Pubblicazione del codice

La richiesta di sviluppare una candidata autorizza il lavoro locale nel perimetro concordato. La richiesta esplicita di pubblicarla può autorizzare branch e PR nominate senza conferma per ogni file. Non autorizza automaticamente force-push, installazione sul Comma, modifica di altre branch o pubblicazione di dati privati.

Prima di un aggiornamento remoto: rileggere il ref, confrontarlo con quello atteso e non sovrascrivere avanzamenti concorrenti. Per una candidata nuova preferire una nuova branch, non una riscrittura della storia condivisa.

Una PR va indirizzata alla baseline che si intende realmente aggiornare. Non fare merge della candidata ricostruita nella vecchia feature solo per “portare tutto dentro”, reintroducendo storia e dipendenze che il procedimento voleva evitare.

### 12.2 Installazione off-road

L’installazione è un’operazione distinta, non un effetto collaterale della skill:

1. identificare veicolo, dispositivo, commit attuale e stato off-road;
2. conservare il pacchetto/commit precedente e verificare come ripristinarlo;
3. selezionare esplicitamente candidata, modello e core coerenti;
4. lasciare NNSLR off al primo avvio;
5. eseguire i controlli previsti dal gate raggiunto;
6. abilitare soltanto la modalità autorizzata e supportata.

La procedura esatta di installazione dipende dal meccanismo effettivamente in uso nel fork e va registrata al primo test target; non viene sostituita da un generico `git pull` sul dispositivo.

### 12.3 Due rollback distinti

**Disabilitazione della feature:** `VisionSpeedLimitMode=off` tramite interfaccia autorizzata. Nessun nuovo lavoro NNSLR; consumer invalidano lo stato. Non è un rimedio garantito a un hang già avvenuto nel driver condiviso, da prevenire con i gate del backend. [P1, §10.5]

**Rollback del software:** ripristinare off-road il precedente insieme verificato di runtime, core e modello. Cambiare solo uno dei tre può rendere il contratto incompatibile. Non riusare un limite salvato prima del riavvio come ipotesi corrente.

Conservare release/tag e ricevute per rollback; non dipendere dal reflog locale dell’agente o dalla cronologia della chat.

<a id="s13"></a>
## 13. Evidenze, rischi e decisioni ancora aperte

### 13.1 Ricevuta obbligatoria per ogni porting

```text
run_id
started_at / finished_at
training_tools_commit
source_feature_version / source_base_commit / source_feature_commit
requested_upstream_ref / resolved_upstream_commit
baseline_gitlinks / candidate_gitlinks
applied_commits / old_to_new_mapping / upstream_satisfied_commits
hook_map_version / structural_adaptations
schema_contract_digest / schema_migrations
core_snapshot_digest / model_bundle_digest
verification_profile / environment_digest
passed_tests / failed_tests / not_run_tests / not_applicable_tests
semantic_comparison / runtime_measurements
privacy_check
result_commit / result_tree
readiness_level / capability_decisions
publication_authorization / deployment_authorization
```

La ricevuta JSON è machine-readable; il report Markdown ne presenta i risultati. I campi relativi a modello/hardware possono risultare non disponibili, con motivazione, senza falsificare un livello superiore di readiness. Gli identificatori personali o delle route vanno esclusi dal report pubblico.

### 13.2 Rischi e trattamento

| Rischio | Trattamento previsto |
|---|---|
| Patch applicata ma hook non più chiamato | Test di invocazione/lifecycle e regressioni, non sola analisi del diff. |
| ID Cap’n Proto riutilizzato | Registro binding e migrazione esplicita con vecchi decoder preservati. |
| Feature off ma import inizializza GPU | Import-isolation e test default-off senza backend. |
| Due agenti sulla stessa candidata | Lock esclusivo, stato persistente, una sola ownership del workspace. |
| Pausa continua per dettagli meccanici | Autonomia di §2.2 e test procedurale R13. |
| Ottimismo sulle V100 trasferito al Comma | Training e target sono ambienti distinti; parity e G4 separati. |
| Dati privati entrano in log, backup o CI | Audit completo, sidecar escluso, test di routing e sanitizzazione. |
| Variante upstream incorpora una funzione simile | Verifica semantica prima di rimuovere/sostituire patch; no duplicazione automatica. |
| Rollback non più possibile | Release precedenti immutabili per procedura, artefatti conservati e prova al banco. |
| Test saltati rappresentati come successo | Ricevuta con esiti separati e gate non stabilito. |

### 13.3 Decisioni non bloccanti per iniziare lo sviluppo

| Decisione | Stato e azione |
|---|---|
| Backend target definitivo | Non stabilito: implementare interfacce e test offline; selezione dopo benchmark autorizzato. |
| Stock o native runner da supportare per primo | Non stabilito sul dispositivo: leggere la configurazione quando autorizzato. |
| Binding dei nuovi messaggi | Da fissare con R03 sulla schema completa; non riservato dalla sola pianificazione. |
| Directory locale degli artefatti e sidecar | Da fissare dopo audit esposizioni; nessun percorso inventato. |
| Modello/capability corrente | Non stabiliti: observation-only è un risultato valido; niente auto-completamento interpretativo. |
| Seconda base reale di riapplicazione | Da selezionare quando disponibile; fixture locali nel frattempo. |
| Destinazione della release e alias stabile | Pubblicazione candidata non distruttiva predefinita; alias deciso separatamente. |

La mancanza di queste decisioni non impedisce R00–R04, test, export del core e motore di riapplicazione su fixture. Non autorizza a dichiarare pronte le parti che dipendono da esse.

<a id="s14"></a>
## 14. Definition of Done e ordine operativo

### 14.1 Mappa di copertura rispetto al v4

| Requisito originale | Task di questo addendum |
|---|---|
| Due repository, core unico, artefatti immutabili (§4.3) | R00–R02 |
| Contratti e publisher (§7) | R03, R06 |
| Tracking, applicabilità, passaggio e continuità (§8) | R07 e dipendenze core originali; nessuna duplicazione runtime |
| Comparator e UI separati (§9) | R07–R08 |
| Backend, scheduling e fault (§10) | R04–R06, R09, R11, R14 |
| Non-interferenza e valutazione (§11) | R10, R14 |
| File map, test e CI (§12) | R00–R15 secondo ownership |
| Gate G0–G7 (§13) | §11 e R14–R15 |
| Nuovo requisito di riapplicazione da master | R01, R12–R13, R15 |

### 14.2 Definition of Done della funzionalità

- [ ] Core e artefatti hanno provenienza e hash verificati; nessun training package installato sul Comma.
- [ ] Sorgenti runtime circoscritti e hook registrati; nessuna scrittura nei campi CAR/MAP o nei controlli.
- [ ] Schema stabile, code limitate, timestamp e unità coerenti.
- [ ] Default off, modalità e capability rispettati, mock esclusi dall’on-road.
- [ ] Test source/build, regressioni, privacy e fault con risultati effettivi.
- [ ] Backend e dominio supportato dichiarati, senza confondere observation e current.
- [ ] G4–G7 soddisfatti soltanto per le capacità che la release realmente abilita.

### 14.3 Definition of Done della riapplicazione

- [ ] Skill caricabile nel client verificato e usabile senza questa conversazione.
- [ ] Script riapplicano una serie `sealed`, non una descrizione incompleta della feature.
- [ ] Upstream corrente fissato per SHA e submoduli rispettati.
- [ ] Stessa base produce stesso albero sorgente e gitlink.
- [ ] Seconda esecuzione non duplica l’integrazione.
- [ ] Spostamenti di hook e conflitti schema coperti da test.
- [ ] Seconda revisione upstream reale verificata, con adattamenti documentati.
- [ ] Nuova candidata, vecchia versione e report rimangono identificabili e recuperabili.
- [ ] Nessuna pubblicazione, riscrittura di ref o installazione implicita.

### 14.4 Sequenza pratica consigliata

```text
R00 baseline
  → R01 catalogo e workspace su fixture
  → R02 core snapshot
  → R03 schema
  → R04 adapter frame/context
  → R05 backend di riferimento
  → R06 lifecycle
  → R07 confronto e capability
  → R08 UI offline + R09 privacy
  → R10 regressioni
  → R12 reapply engine + R13 skill
  → R11 solo se il percorso Chestnut viene selezionato
  → R14 prove target/shadow autorizzate
  → R15 seconda base, release e rollback
```

R12/R13 possono progredire in parallelo ai moduli runtime usando fixture Git locali; la sigillatura della prima serie reale avviene solo quando esiste il relativo codice. L’UI può essere sviluppata con fixture mentre training e interpretazione proseguono, ma questo non abilita la UI sulla vettura.

**Primo risultato utile:** un candidato source-ready con core/advisory separati e un porting ripetibile su fixture; successivamente una release observation-only misurata. Non attendere di risolvere ogni semantica stradale per rendere utili osservazione, replay e manutenzione.

**Criterio finale:** poter chiedere «riapplica questa versione NNSLR sul master Sunnypilot corrente» e ottenere codice, test, diff, capacità e limiti verificabili, mantenendo intatta la versione funzionante. Non promettere che qualunque cambiamento upstream sarà sempre compatibile o risolvibile senza review.

<a id="s15"></a>
## 15. Fonti e tracciabilità

Consultazione: 18 settembre 2026. Le fonti del progetto sono state lette tramite il connettore GitHub; i riferimenti ai file usano revisioni immutabili. I ref di branch riportati in §1 fotografano quella consultazione, non il loro stato futuro.

### Piano e baseline

- **[P1]** [Piano v4, `docs/plan.md`, TRAIN_REPO alla revisione verificata](https://github.com/cristianku/openpilot-nnslr-tools/blob/9ffa00057905025c9fe6c8226664acc52330c32d/docs/plan.md). In particolare: vincoli globali; §4.3 ownership/core; §§7–9 protocollo e semantica; §10 backend/Chestnut/privacy; §11 test; §§12–13 task e gate. Il percorso runtime storico del v4 è aggiornato esplicitamente in questo addendum, non nascosto.
- **[B1]** [Commit TRAIN_REPO `9ffa000…`](https://github.com/cristianku/openpilot-nnslr-tools/commit/9ffa00057905025c9fe6c8226664acc52330c32d). Ref `main` osservato a questa revisione.
- **[B2]** [Commit fork Sunnypilot `a5f44653…`](https://github.com/cristianku/sunnypilot/commit/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56). Ref `nn-speed-limit-vision` osservato a questa revisione.
- **[B3]** [Commit upstream Sunnypilot `a5f44653…`](https://github.com/sunnypilot/sunnypilot/commit/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56). Ref `master` osservato a questa revisione.

### Sorgente runtime verificato

- **[R1]** [`modeld.py`, proprietà di Chestnut](https://github.com/cristianku/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/selfdrive/modeld/modeld.py#L78-L110).
- **[R2]** [`process_config.py`, selezione runner e processi](https://github.com/cristianku/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/system/manager/process_config.py).
- **[R3]** [`custom.capnp`, schema operativo e struct riservate](https://github.com/cristianku/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/cereal/custom.capnp).
- **[R4]** [`UIStateSP`, subscription estese](https://github.com/cristianku/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/openpilot/selfdrive/ui/sunnypilot/ui_state.py#L30-L40).
- **[R5]** [`.gitmodules`, incluso `neural_network_data`](https://github.com/cristianku/sunnypilot/blob/a5f44653d7f43ad57fef2f546f3916ec4cbf3c56/.gitmodules).

### Documentazione esterna primaria

- **[E1]** [Git — `cherry-pick`](https://git-scm.com/docs/git-cherry-pick): applicazione dei commit e gestione dei conflitti.
- **[E2]** [Git — `range-diff`](https://git-scm.com/docs/git-range-diff): confronto tra versioni di una serie di patch.
- **[E3]** [Cap’n Proto — Schema language, Evolving Your Protocol](https://capnproto.org/language.html#evolving-your-protocol): regole di compatibilità di tipi e ordinali.
- **[E4]** [Git — `worktree`, sezione BUGS](https://git-scm.com/docs/git-worktree#_bugs): limitazioni documentate dei submoduli nei checkout multipli.
- **[E5]** [Git — `submodule`](https://git-scm.com/docs/git-submodule): differenza fra gitlink fissati e `update --remote`.
- **[E6]** [GitHub — Adding agent skills for GitHub Copilot](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills): struttura e collocazione di `SKILL.md`.
- **[E7]** [GitHub — Adding agent skills for Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills): caricamento, verifica e invocazione della skill nel client.

Non vengono affermate regole giuridiche svizzere ulteriori rispetto alle astensioni del piano v4. L’interpretazione di zone, cancellazioni, pannelli o limiti generali richiederà la review delle fonti ufficiali nel task dedicato, prima di abilitare tali capability.

---

**Fine del piano.** I checkbox sono inizialmente non completati. La prima esecuzione deve partire dallo stato realmente trovato nei repository e riportare separatamente ciò che era già presente, ciò che viene implementato e ciò che non è stato ancora verificato.