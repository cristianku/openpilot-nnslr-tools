# Codex — Configurazione automatica del sub-agent locale `llama-v100`

## Obiettivo

Configura **autonomamente** Codex affinché possa usare il mio server locale `llama-v100` come sub-agent globale, indipendentemente dal workspace/repository aperto.

Non chiedermi di modificare file manualmente.

Devi:

1. ispezionare la configurazione Codex esistente;
2. fare backup prima di ogni modifica;
3. configurare il provider locale senza rompere le impostazioni esistenti;
4. creare un agente globale `llama_v100`;
5. aggiungere una policy globale in `~/.codex/AGENTS.md`;
6. verificare che Codex stia realmente inviando il lavoro al server V100;
7. se il meccanismo nativo dei sub-agent con custom provider non funziona, rilevarlo automaticamente e implementare un fallback funzionante;
8. lasciare la configurazione finale persistente per tutti i workspace.

---

# Informazioni sul server locale

Il server locale è già attivo qui:

```text
http://10.10.10.134:8080
```

Modello:

```text
/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
```

Nome logico:

```text
llama-v100
```

Il server è basato su `llama.cpp`.

La configurazione attualmente usata da VS Code/custom endpoint è:

```json
[
  {
    "name": "llama-v100",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
        "name": "Qwen 3.8 27B",
        "url": "http://10.10.10.134:8080/v1/chat/completions",
        "toolCalling": true,
        "vision": true,
        "thinking": true,
        "supportsReasoningEffort": [
          "low",
          "medium",
          "xhigh"
        ],
        "reasoningEffortFormat": "chat-completions",
        "contextWindow": 139264,
        "maxOutputTokens": 16384
      }
    ],
    "settings": {
      "/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf": {
        "reasoningEffort": "low"
      }
    }
  }
]
```

Questa configurazione è solo informativa.

**Non modificarla e non dipendere da essa.**

La configurazione Codex deve essere indipendente.

---

# Regole operative

## Non chiedere interventi manuali

Non dirmi:

- "aggiungi questo a config.toml";
- "crea questo file";
- "esegui questo comando";
- "controlla tu i log";
- "riavvia tu Codex".

Devi fare automaticamente tutto ciò che puoi fare dal sistema.

Se un riavvio di Codex è strettamente necessario per caricare una nuova configurazione globale, completa comunque prima tutte le modifiche e i test possibili e spiegami alla fine soltanto cosa richiede una nuova sessione.

---

# Fase 1 — Rileva il Codex home effettivo

Non assumere ciecamente `~/.codex`.

Controlla:

```bash
printf 'CODEX_HOME=%s\n' "${CODEX_HOME:-}"
printf 'HOME=%s\n' "$HOME"
```

Se `CODEX_HOME` è valorizzato:

```text
CODEX_DIR=$CODEX_HOME
```

altrimenti:

```text
CODEX_DIR=$HOME/.codex
```

Crea soltanto le directory mancanti:

```bash
mkdir -p "$CODEX_DIR"
mkdir -p "$CODEX_DIR/agents"
```

Da questo punto usa sempre il vero `$CODEX_DIR`.

---

# Fase 2 — Ispeziona la configurazione corrente

Prima di modificare qualsiasi cosa:

```bash
codex --version || true
ls -la "$CODEX_DIR" || true
```

Leggi, se presenti:

```text
$CODEX_DIR/config.toml
$CODEX_DIR/AGENTS.md
$CODEX_DIR/AGENTS.override.md
$CODEX_DIR/agents/
```

Devi preservare tutte le impostazioni esistenti.

Non sostituire completamente `config.toml`.

Non cancellare provider, profili, MCP, sandbox settings, modelli o altre configurazioni già presenti.

---

# Fase 3 — Backup

Prima della prima modifica crea una directory di backup timestampata.

Esempio:

```bash
BACKUP_DIR="$CODEX_DIR/backup-before-llama-v100-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
```

Copia, se presenti:

```bash
cp -a "$CODEX_DIR/config.toml" "$BACKUP_DIR/" 2>/dev/null || true
cp -a "$CODEX_DIR/AGENTS.md" "$BACKUP_DIR/" 2>/dev/null || true
cp -a "$CODEX_DIR/AGENTS.override.md" "$BACKUP_DIR/" 2>/dev/null || true
cp -a "$CODEX_DIR/agents" "$BACKUP_DIR/" 2>/dev/null || true
```

Mostra alla fine dove si trova il backup.

---

# Fase 4 — Verifica il server llama.cpp

Prima prova la raggiungibilità:

```bash
curl -fsS --max-time 5 http://10.10.10.134:8080/health
```

Se `/health` non esiste o restituisce un formato differente, non considerarlo automaticamente un errore.

Verifica almeno:

```bash
curl -fsS --max-time 10 http://10.10.10.134:8080/v1/models
```

Poi verifica la Responses API, che è quella preferita per il provider Codex:

```bash
curl -fsS --max-time 120 \
  http://10.10.10.134:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer no-key' \
  -d '{
    "model": "/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
    "input": "Reply with exactly: LLAMA_V100_RESPONSES_OK"
  }'
```

Controlla realmente che la risposta contenga:

```text
LLAMA_V100_RESPONSES_OK
```

Se `/v1/responses` funziona, usa:

```text
wire_api = "responses"
```

Se `/v1/responses` non funziona ma `/v1/chat/completions` funziona, non inventare configurazioni non supportate.

In quel caso:

1. identifica versione/build di `llama.cpp`, se possibile;
2. determina se il server va aggiornato;
3. non rompere il server esistente;
4. preferisci un fallback compatibile descritto più avanti.

Verifica inoltre chat completions come controllo secondario:

```bash
curl -fsS --max-time 120 \
  http://10.10.10.134:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer no-key' \
  -d '{
    "model": "/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: LLAMA_V100_CHAT_OK"
      }
    ],
    "temperature": 0
  }'
```

---

# Fase 5 — Configura il provider Codex

Il provider deve chiamarsi:

```text
llama_v100
```

Il blocco desiderato è concettualmente:

```toml
[model_providers.llama_v100]
name = "llama-v100"
base_url = "http://10.10.10.134:8080/v1"
wire_api = "responses"
requires_openai_auth = false
```

Se la versione corrente di Codex richiede un bearer token dummy per custom provider locali, usa soltanto il meccanismo supportato dalla versione installata.

Non aggiungere segreti reali.

Non modificare il provider principale della sessione Codex.

Il modello principale deve continuare a essere quello configurato dall'utente/OpenAI.

`llama_v100` serve come worker secondario.

## Modifica sicura del TOML

Non usare una semplice append cieca se il blocco esiste già.

Devi:

1. analizzare il TOML esistente;
2. verificare se `[model_providers.llama_v100]` esiste;
3. se non esiste, aggiungerlo;
4. se esiste, aggiornare soltanto le chiavi necessarie;
5. mantenere intatto tutto il resto.

Se hai Python disponibile, puoi usare un parser TOML affidabile.

Se non è disponibile un writer TOML sicuro, effettua una modifica testuale molto conservativa e verifica successivamente il file.

Dopo la modifica verifica che Codex riesca almeno a leggere la configurazione:

```bash
codex --version
```

e, se disponibile nella versione installata, usa anche un comando di config/debug appropriato.

---

# Fase 6 — Crea il custom sub-agent globale

Crea:

```text
$CODEX_DIR/agents/llama-v100.toml
```

Deve essere un agente personale/globale, non specifico del repository.

Configurazione desiderata:

```toml
name = "llama_v100"

description = """
Local Qwen3.8 27B worker running on a Tesla V100.
Use this agent for codebase exploration, code search, call-path tracing,
large-file analysis, repetitive investigation, log analysis, test-failure
analysis, comparison tasks, and evidence gathering before implementation.
"""

model = "/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"
model_provider = "llama_v100"
model_reasoning_effort = "low"
sandbox_mode = "read-only"

developer_instructions = """
You are a local worker assisting a stronger parent Codex agent.

Your purpose is to offload exploratory, mechanical, repetitive, and
context-heavy work from the parent agent.

Prefer tasks such as:

- locating files, symbols, classes, functions, constants and configuration;
- searching usages and references;
- tracing execution paths and dependencies;
- understanding unfamiliar code;
- summarizing large files or groups of files;
- comparing implementations;
- inspecting logs;
- analyzing test failures;
- identifying likely files affected by a requested change;
- gathering concrete evidence required by the parent agent;
- reviewing code when explicitly requested.

When reporting findings:

- be concise;
- give exact file paths;
- give exact symbol/function/class names;
- provide line numbers where practical;
- separate verified facts from hypotheses;
- state uncertainty explicitly;
- do not invent unavailable information.

Do not make final architectural decisions unless explicitly requested.

Do not modify files unless the parent explicitly delegates an implementation
task and the sandbox permits it.

The parent Codex agent remains responsible for final planning, correctness,
architectural decisions, implementation decisions, and validation.
"""
```

Dopo aver scritto il file, rileggilo e verifica che sia TOML valido.

---

# Fase 7 — Aggiungi la policy globale a AGENTS.md

Il comportamento deve essere valido indipendentemente dal workspace.

Il file globale è:

```text
$CODEX_DIR/AGENTS.md
```

## Importante: AGENTS.override.md

Se esiste:

```text
$CODEX_DIR/AGENTS.override.md
```

Codex può usare quello al posto di `AGENTS.md` a livello globale.

Quindi:

1. rileva se esiste `AGENTS.override.md`;
2. non cancellarlo;
3. se è un override permanente dell'utente, integra la policy nel file globale effettivamente caricato;
4. se sembra temporaneo, non distruggerlo: segnala chiaramente la precedenza;
5. verifica quale file verrà effettivamente caricato.

## Inserimento idempotente

Nel file globale effettivamente utilizzato inserisci un blocco delimitato:

```markdown
<!-- BEGIN LLAMA_V100_GLOBAL_POLICY -->

## Local V100 sub-agent delegation

A local sub-agent named `llama_v100` is available.

It runs Qwen3.8 27B locally on a Tesla V100 and should be used to offload
exploratory, repetitive, and context-heavy work from the primary Codex model.

### When to delegate

Proactively delegate to `llama_v100` when useful for:

- repository exploration;
- finding files, symbols, classes and functions;
- searching usages and references;
- tracing call paths and execution flow;
- understanding unfamiliar code;
- summarizing large files or related groups of files;
- comparing implementations;
- inspecting logs;
- analyzing test failures;
- identifying files affected by a requested change;
- collecting evidence before implementation;
- repetitive analysis that does not require the strongest reasoning model.

### Preferred workflow

For substantial coding tasks:

1. Decide whether exploratory work is required.
2. If yes, delegate a clearly bounded task to `llama_v100`.
3. Give the sub-agent a precise objective and expected output.
4. Wait for its findings.
5. Independently verify important findings.
6. Use the findings for planning and implementation.

The primary Codex model remains responsible for:

- final architecture;
- ambiguous or high-impact decisions;
- security-sensitive conclusions;
- final implementation decisions;
- verification of important claims;
- final testing and validation.

Do not blindly trust the local agent.

Do not duplicate work already performed by `llama_v100` unless verification
is necessary.

There is currently only one V100 worker. Do not start multiple
`llama_v100` jobs concurrently.

If `llama_v100` is unavailable or fails, continue with the primary model
instead of blocking the user's task.

Use the local worker proactively where it can reduce primary-model work,
but do not delegate purely for the sake of delegation.

<!-- END LLAMA_V100_GLOBAL_POLICY -->
```

Se il blocco esiste già:

- aggiornalo;
- non duplicarlo.

Preserva integralmente tutte le altre istruzioni globali dell'utente.

---

# Fase 8 — Verifica che Codex carichi la policy globale

Dopo le modifiche esegui, se compatibile con la versione installata:

```bash
codex --ask-for-approval never "Summarize the global instructions you received about llama_v100. Do not modify files."
```

La risposta deve dimostrare che Codex vede la policy globale.

Se il comando/versione usa flag differenti, usa l'equivalente supportato.

---

# Fase 9 — Test diretto del provider locale tramite Codex

Prima di testare il sub-agent, verifica il provider come sessione diretta.

Usa un comando equivalente a:

```bash
codex \
  -c model_provider='"llama_v100"' \
  -c model='"/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"' \
  -c model_reasoning_effort='"low"' \
  "Reply with exactly: CODEX_LLAMA_V100_DIRECT_OK"
```

Adatta correttamente la sintassi alla versione installata.

Il test è riuscito soltanto se:

1. Codex termina senza errore;
2. la risposta contiene `CODEX_LLAMA_V100_DIRECT_OK`;
3. la richiesta è realmente arrivata al server `10.10.10.134:8080`.

Non considerare sufficiente che il modello risponda: devi evitare che il test cada accidentalmente sul provider OpenAI.

Se possibile, verifica contemporaneamente tramite:

- log di `llama-server`;
- metriche endpoint;
- access log;
- timestamp delle richieste;
- output verbose/debug Codex.

---

# Fase 10 — Test del vero sub-agent

Avvia una sessione Codex con il provider principale normale e chiedi esplicitamente:

```text
Spawn the custom agent `llama_v100`.

Give it this exact task:

"Inspect the current working directory, do not modify anything, and return:
LLAMA_V100_SUBAGENT_OK
followed by the absolute working directory and one visible filename."

Do not perform the delegated investigation yourself.
Wait for the sub-agent result and return it.
```

Il test è riuscito solo se hai prova che la richiesta del child agent è stata processata da:

```text
10.10.10.134:8080
```

Non basta vedere il nome `llama_v100` nell'interfaccia.

Devi verificare il routing reale.

---

# Fase 11 — Gestisci il bug noto dei custom-provider sub-agent

Esistono versioni di Codex in cui un custom sub-agent può:

- ereditare erroneamente il provider del parent;
- ignorare `model_provider`;
- non ricevere correttamente il task dinamico;
- non esporre correttamente `spawn_agent` con un custom provider.

Per questo motivo **non assumere che la configurazione nativa funzioni solo perché il TOML è valido**.

Verifica il comportamento effettivo.

Se il test nativo fallisce ma il test diretto del provider locale funziona, implementa automaticamente il fallback seguente.

---

# Fase 12 — Fallback robusto: local worker tramite comando

Se il routing nativo del custom sub-agent è rotto nella versione installata, crea un wrapper globale.

Percorso suggerito:

```text
$CODEX_DIR/bin/llama-v100-agent
```

Crea anche la directory:

```bash
mkdir -p "$CODEX_DIR/bin"
```

Il wrapper deve:

1. ricevere il task come argomento o stdin;
2. avviare una sessione Codex separata;
3. forzare:
   - `model_provider = llama_v100`;
   - modello Qwen locale;
   - reasoning low;
4. impedire che cada sul provider OpenAI;
5. restituire stdout al parent;
6. propagare correttamente exit code ed errori.

Schema concettuale:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL='/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf'

if [[ $# -gt 0 ]]; then
  TASK="$*"
else
  TASK="$(cat)"
fi

exec codex exec \
  -c model_provider='"llama_v100"' \
  -c model="\"$MODEL\"" \
  -c model_reasoning_effort='"low"' \
  "$TASK"
```

**Non copiare alla cieca questa sintassi.**

Prima controlla:

```bash
codex exec --help
```

e usa la sintassi esatta supportata dalla versione corrente.

Rendi eseguibile:

```bash
chmod 0755 "$CODEX_DIR/bin/llama-v100-agent"
```

Poi testalo con:

```text
Reply exactly: LLAMA_V100_WRAPPER_OK
```

e verifica di nuovo che la richiesta arrivi fisicamente al server V100.

---

# Fase 13 — Se viene usato il fallback, aggiorna la policy globale

Se il sub-agent nativo non può usare correttamente il custom provider, aggiorna il blocco globale `LLAMA_V100_GLOBAL_POLICY`.

La policy deve dire al Codex principale che, per delegare lavoro locale, deve usare:

```text
$CODEX_DIR/bin/llama-v100-agent
```

passando un task completo e autosufficiente.

Esempio concettuale:

```bash
"$CODEX_DIR/bin/llama-v100-agent" \
  "Inspect this repository and locate the implementation of X.
   Do not modify files.
   Return exact paths, symbols and relevant findings."
```

## Importante sul contesto

Il worker separato non deve ricevere task vaghi come:

```text
look at that function
```

Deve ricevere un prompt autosufficiente contenente almeno:

- obiettivo;
- repository/current working directory;
- file o simboli noti;
- restrizioni;
- output atteso.

Quando viene invocato da un repository, eseguilo con il working directory corretto affinché il worker possa vedere lo stesso codebase.

---

# Fase 14 — Policy di utilizzo intelligente

Non usare `llama_v100` per qualsiasi micro-task.

Usalo soprattutto per lavoro ad alto volume e relativamente meccanico.

## Ottimo per il V100 locale

- scandire repository;
- trovare simboli;
- grep/search semantico assistito;
- leggere molti file;
- ricostruire execution flow;
- individuare dipendenze;
- analizzare log;
- analizzare test failure;
- confrontare implementazioni;
- preparare riassunti tecnici;
- individuare file candidati a una modifica;
- review preliminare;
- raccogliere evidenze.

## Mantieni sul Codex principale

- pianificazione finale;
- decisioni architetturali;
- modifiche ad alto impatto;
- problemi molto ambigui;
- security-sensitive reasoning;
- validazione definitiva;
- integrazione delle conclusioni provenienti da più fonti;
- decisione finale su quali modifiche applicare.

---

# Fase 15 — Nessuna concorrenza per ora

Attualmente esiste un solo server:

```text
llama-v100
10.10.10.134:8080
```

Quindi limita il parallelismo di questo specifico worker a **1 task alla volta**.

Non creare due richieste contemporanee allo stesso worker salvo che il server sia esplicitamente configurato e testato per farlo.

Non ridurre globalmente il parallelismo degli altri eventuali agenti Codex se non è necessario.

L'obiettivo è limitare il worker V100, non l'intero sistema multi-agent.

---

# Fase 16 — Preparazione per una futura seconda V100

Non configurare ora un secondo agente inesistente.

Però struttura i nomi in modo che in futuro sia facile aggiungere:

```text
llama_v100_1
llama_v100_2
```

oppure:

```text
v100_explorer
v100_reviewer
```

Quando esisterà una seconda V100, l'architettura desiderata sarà:

```text
Primary Codex
   |
   +--> local V100 agent #1 -> exploration / repository analysis
   |
   +--> local V100 agent #2 -> testing / review / independent analysis
```

I due worker dovranno essere indipendenti e potranno lavorare realmente in parallelo.

Non implementare questa parte finché non esiste il secondo endpoint.

---

# Fase 17 — Verifiche finali obbligatorie

Prima di dichiarare completato il lavoro verifica:

## Configurazione

```text
[ ] Codex home reale individuato
[ ] backup creato
[ ] config.toml preservato
[ ] provider llama_v100 presente
[ ] AGENTS.md globale preservato
[ ] policy llama_v100 presente una sola volta
[ ] custom agent globale presente
```

## Server

```text
[ ] 10.10.10.134:8080 raggiungibile
[ ] /v1/models verificato
[ ] /v1/responses verificato oppure incompatibilità documentata
[ ] risposta realmente generata da Qwen locale
```

## Codex

```text
[ ] provider locale testato direttamente
[ ] policy globale caricata
[ ] sub-agent nativo testato
[ ] routing reale verso V100 verificato
```

## Fallback

```text
[ ] non necessario, oppure
[ ] wrapper creato
[ ] wrapper testato
[ ] policy globale aggiornata per usarlo
[ ] routing del wrapper verificato
```

---

# Fase 18 — Report finale

Alla fine dammi un report breve con questo formato:

```text
Codex local V100 integration

Codex version:
Codex home:

llama.cpp endpoint:
Model:

Responses API:
Direct Codex provider test:
Native custom sub-agent:
Fallback wrapper:

Global agent file:
Global policy file:
Backup:

Final status:
```

Per `Final status` usa uno dei seguenti valori:

```text
WORKING — native sub-agent
WORKING — fallback local worker
PARTIAL — direct local provider only
FAILED — local endpoint unavailable
FAILED — Codex incompatibility
```

Aggiungi poi massimo 5 righe con eventuali problemi trovati.

---

# Criterio fondamentale

Non considerare il lavoro concluso perché i file di configurazione "sembrano corretti".

Il risultato richiesto è:

```text
Codex principale
       |
       | delega task
       v
Qwen3.8 27B
Tesla V100
10.10.10.134:8080
```

e deve esserci una verifica concreta che la richiesta sia arrivata al server locale.

---

# Comportamento permanente desiderato

Una volta terminata la configurazione, nelle sessioni future Codex deve ricordarsi automaticamente:

> Ho a disposizione un worker locale `llama_v100`. Per exploration, repository scanning, ricerca di simboli, analisi ripetitiva, log, test failure e raccolta di evidenze dovrei delegare a lui quando è utile. Io rimango responsabile della pianificazione, delle decisioni importanti e della validazione finale.

Questa regola deve essere disponibile **in qualunque workspace**, non soltanto nel repository in cui viene effettuata questa configurazione.

---

# Riferimenti tecnici verificati

OpenAI Codex — AGENTS.md:
https://developers.openai.com/docs/agent-configuration/agents-md

OpenAI Codex — subagents/custom agents:
https://developers.openai.com/docs/agent-configuration/subagents

llama.cpp server — OpenAI-compatible Responses API:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Codex issue — native subagent can ignore explicit custom `model_provider`:
https://github.com/openai/codex/issues/40858

Codex issue — custom-provider subagent dynamic task payload:
https://github.com/openai/codex/issues/35932
