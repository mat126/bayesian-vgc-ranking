# Ranking VGC — Bradley-Terry gerarchico bayesiano

Modello di ranking per il metagame competitivo Pokémon VGC, costruito per stimare quali
Pokémon e quali squadre arrivano da favoriti ai Campionati Mondiali 2026 (Pokémon Champions,
Regulation Set M-B).

La forza di una squadra non viene stimata come parametro libero, cosa impossibile data la
combinatoria, ma **decomposta negli effetti delle sue componenti**: specie, mosse, abilità,
strumenti e nature, più un effetto di abilità del giocatore. L'inferenza è bayesiana
gerarchica, con partial pooling che regolarizza i livelli rari e intervalli di credibilità
su ogni stima.

> **Stato attuale.** I risultati riportati sotto provengono da inferenza variazionale (ADVI),
> veloce ma che approssima il posterior con una gaussiana fattorizzata e tende a
> **sottostimare l'incertezza**. Gli intervalli vanno letti come ottimistici. La stima
> definitiva con NUTS è in corso: vedi [Roadmap](#roadmap).

---

## Risultati principali

Dataset: **17.581 match** da tornei Limitless in Regulation M-B, con teamlist completa per
entrambi i giocatori, 3.878 giocatori distinti, 7.025 righe squadra-torneo.

### Il pilota conta più della squadra

La scala `tau` di ciascun blocco misura quanto quella dimensione muove la forza. Il risultato
è netto e va letto prima di tutto il resto:

| Blocco | tau (media) | 5%–95% |
|---|---|---|
| **giocatore** | **1.074** | 1.052 – 1.096 |
| strumento | 0.213 | 0.192 – 0.235 |
| specie | 0.201 | 0.184 – 0.219 |
| abilità | 0.194 | 0.176 – 0.213 |
| mosse | 0.187 | 0.175 – 0.199 |
| natura | 0.121 | 0.088 – 0.158 |

L'abilità del pilota ha dispersione **circa cinque volte** superiore a qualunque
caratteristica della squadra. Detto in modo diretto: in questi dati chi gioca conta molto più
di cosa gioca. Fra le feature di squadra, strumento e specie guidano, la natura è marginale.

**Caveat di identificabilità, importante.** L'effetto pilota e la forza della squadra si
separano solo grazie all'incrocio giocatori × squadre nei dati. Un giocatore che porta sempre
la stessa lista rende le due quantità parzialmente collineari, e parte di quel `tau = 1.07`
può essere forza-squadra assorbita dal termine del pilota. La gerarchia dei blocchi va quindi
letta come indicativa, non come stima causale pulita.

### Capacità predittiva (held-out)

| Metrica | Modello | Baseline |
|---|---|---|
| log-loss (test) | **0.6618** | 0.6931 (coin flip) |
| accuracy (test) | **0.6141** | 0.5 |

Il segnale c'è ed è reale, ma modesto: il modello batte il caso in modo consistente senza
avvicinarsi a una previsione affidabile del singolo match. È un esito atteso in un gioco con
forte componente stocastica e decisioni in-battle non osservate. Nota inoltre che questa
performance **include** l'effetto pilota, che secondo le `tau` è il predittore dominante:
la quota attribuibile alla sola composizione della squadra è di conseguenza più bassa.

### Ranking dei Pokémon

Effetto specie `beta_species`, cioè il contributo alla forza a parità di tutto il resto.
La classifica va letta **per intervallo, non per media**: diverse specie in cima hanno il
5° percentile sotto zero, quindi la loro posizione è guidata dal rumore. Le stime robuste
sono quelle con intervallo interamente positivo.

| Specie | media | 5%–95% | robusta |
|---|---|---|---|
| Armarouge | 0.255 | −0.007 – 0.536 | no |
| Gallade | 0.216 | 0.013 – 0.420 | al limite |
| **Charizard** | 0.202 | 0.103 – 0.294 | **sì** |
| Gardevoir | 0.199 | 0.043 – 0.362 | sì |
| Froslass | 0.199 | 0.081 – 0.327 | sì |
| Wash Rotom | 0.174 | 0.051 – 0.297 | sì |
| Klefki | 0.173 | −0.128 – 0.475 | no |
| **Eternal Flower Floette** | 0.162 | 0.060 – 0.251 | **sì** |
| Whimsicott | 0.158 | 0.056 – 0.260 | sì |

Charizard ed Eternal Flower Floette sono le stime più affidabili del gruppo di testa, con
intervalli stretti e interamente positivi.

### Effetti di set

Le Mega Stone dominano la classifica degli strumenti, come atteso in un formato con Mega
Evoluzioni e senza Terastallizzazione: **Venusaurite** (0.368, il valore più alto in assoluto
fra tutte le feature), Floettite (0.206), Raichunite Y (0.197), Dragoninite (0.187).

Fra le mosse emergono controllo e utility più che potenza bruta: Quick Guard (0.251),
Perish Song (0.237), Snarl (0.229), Encore (0.202), Weather Ball (0.202).

Fra le nature spicca **Calm** (0.184), l'unica nettamente separata da zero, coerente con la
lettura della natura come proxy dell'archetipo di build difensivo-speciale.

### Squadre favorite

Su **5.393 squadre uniche** solo **95** compaiono almeno 5 volte: conferma empirica che
stimare una forza libera per squadra sarebbe stato impossibile, e che la decomposizione in
feature è necessaria e non una raffinatezza. Le prime posizioni sono occupate da nuclei
Sneasler / Kingambit / Incineroar / Sinistcha e da varianti Charizard / Aerodactyl / Farigiraf.

Testa a testa fra le prime due, a parità di pilota: **P(A batte B) = 0.547 [0.476, 0.622]**,
cioè un vantaggio reale ma non decisivo, con intervallo che sfiora il pareggio.

Figure in [`figures/`](figures/), tabelle complete in [`results/`](results/).

---

## Struttura del repository

```
ranking-vgc/
├── README.md
├── environment.yml           ambiente conda riproducibile
├── .gitignore
├── src/                      tutto il codice, importabile fra moduli
│   ├── limitless_vgc.py      raccolta match dall'API Limitless
│   ├── pokepaste.py          parser pokepaste (fallback per fonti non-API)
│   ├── inspect_standings.py  diagnostica struttura API
│   ├── showdown_replays.py   fonte secondaria (ladder, solo specie)
│   ├── canonicalize.py       pulizia campi (multilingua, refusi, nulli)
│   ├── design_matrix.py      matrici conteggio-differenza e conteggi per lato
│   ├── bt_bayes.py           modello bayesiano gerarchico (PyMC)
│   ├── neural_bt.py          architettura Set Transformer siamese
│   ├── neural_data.py        encoding in tensori, vocabolari, DataLoader
│   ├── neural_train.py       training con early stopping
│   └── neural_eval.py        metriche held-out, ranking, MC-dropout
├── notebooks/
│   └── analisi_vgc.ipynb     analisi e figure dal posterior salvato
├── docs/
│   └── architettura_neural_bt.svg
├── figures/                  grafici versionati
├── results/                  CSV dei ranking versionati
├── data/                     [non versionato] rigenerabile dalla pipeline
└── models/                   [non versionato] posterior e checkpoint
```

Gli script stanno tutti in `src/` piatto, così gli import fra moduli funzionano senza
pacchetti né path hack: lanciandoli come `python src/nome.py` dalla radice, Python mette
`src/` in testa al path e i percorsi dati relativi si risolvono dalla radice del progetto.

---

## Pipeline

### 1. Ambiente

```bash
conda env create -f environment.yml
conda activate vgc
```

### 2. Raccolta dati

Fonte primaria: **API pubblica Limitless**, senza chiave. Le usage stats di Victory Road o
Pikalytics non bastano: per un Bradley-Terry servono esiti testa a testa con entrambe le
squadre note. L'endpoint `standings` fornisce le teamlist già strutturate e `pairings` i
vincitori; il join dà (squadra A, squadra B, esito).

```bash
python src/limitless_vgc.py --list-formats
python src/limitless_vgc.py --format <FORMAT_ID_REG_M_B> --min-players 30 --max-tournaments 300
```

Produce `data/matches.jsonl` e `data/teams.jsonl`.

### 3. Pulizia

I campi testuali arrivano sporchi perché ogni giocatore invia la lista nella lingua del
proprio gioco, con refusi e maiuscole incoerenti. Il vocabolario però è chiuso e noto, quindi
la canonicalizzazione è quasi interamente automatica: slug, aggancio esatto alle tabelle
multilingua PokéAPI, fuzzy match sui refusi, override manuali per il residuo.

```bash
python src/canonicalize.py data/matches.jsonl data/matches_clean.jsonl --fuzz 85
```

Le Mega Stone introdotte da Champions e assenti da PokéAPI (Floettite, Raichunite,
Staraptite e altre 49) vengono riconosciute e **mantenute** col nome grezzo, che è pulito e
consistente: sono feature ad alto valore, non rumore.

### 4. Modello bayesiano

```bash
# veloce, per iterare
python src/bt_bayes.py --matches data/matches_clean.jsonl --method advi --draws 800 --min-freq 5

# con sinergia specie×specie a rango basso
python src/bt_bayes.py --matches data/matches_clean.jsonl --method advi --draws 800 --min-freq 5 --synergy-dim 4

# stime definitive (lento)
python src/bt_bayes.py --matches data/matches_clean.jsonl --method nuts --draws 800 --min-freq 5
```

Utile in fase di test: `--limit N` campiona N match e riduce drasticamente i tempi di
compilazione del grafo.

### 5. Analisi

```bash
jupyter lab notebooks/analisi_vgc.ipynb
```

Il notebook legge il posterior salvato e produce classifiche, figure e la validazione
held-out. Va eseguito con la radice del progetto come working directory.

### 6. Modello neurale (alternativo)

```bash
python src/neural_train.py --matches data/matches_clean.jsonl --epochs 60
python src/neural_eval.py --ckpt models/neural_ckpt.pt --matches data/matches_clean.jsonl --teams data/teams.jsonl
```

---

## Note di modello

**Verosimiglianza.** Bradley-Terry Bernoulli: `logit P(A batte B) = s(A) − s(B)`, con la
struttura a differenza che impone l'antisimmetria per costruzione. Un'intercetta di lato
assorbe l'asimmetria sistematica osservata (frequenza di vittoria di player1 pari a 0.513).

**Forza strutturata.** `s(T)` è somma dei contributi dei sei Pokémon, ciascuno somma degli
effetti delle sue feature. Essendo lineare nei conteggi, la differenza di forze diventa il
prodotto dei coefficienti per la differenza dei vettori-conteggio delle due squadre: da qui
la design matrix.

**Identificabilità.** Ogni squadra ha esattamente sei Pokémon, quindi una costante additiva
per blocco si cancella nella differenza. Il vincolo somma-zero (`ZeroSumNormal`) rimuove il
grado di libertà spurio, ed essendo già in parametrizzazione non centrata evita la geometria
a imbuto tipica dei modelli gerarchici.

**Osservazione parziale.** Le abilità mancano in circa 8.500 slot. Il partial pooling gestisce
il caso senza scartare righe: le feature non osservate marginalizzano al prior e il match
contribuisce comunque tramite gli altri blocchi.

**Sinergia.** Termine opzionale a rango basso (Factorization Machine): un vettore latente per
specie, sinergia di coppia come prodotto scalare, calcolata in forma chiusa. Essendo
quadratica richiede i conteggi **per lato** e non la sola differenza. Numericamente delicata
in ADVI, stabilizzata con prior stretto, clipping del predittore e learning rate ridotto.

**Scelta delle feature.** Gli spread EV/IV sono esclusi per disponibilità e dimensionalità.
Il costo è varianza, non distorsione, dato che l'obiettivo è il ranking e non l'effetto
causale dello spread. La natura funge da proxy parziale dell'archetipo di build.

---

## Limiti noti

**Transitività stocastica.** Una scala di forza unidimensionale non può rappresentare i
triangoli di matchup, che nel VGC esistono. Se la validazione mostrasse errore concentrato su
accoppiamenti specifici servirebbe un termine di matchup antisimmetrico a rango basso.

**Confondimento pilota.** Vedi il caveat sopra. È il limite più rilevante per l'interpretazione
dei risultati.

**Stazionarietà.** Il modello è una fotografia della finestra M-B. Il meta si sposta di
settimana in settimana e i risultati vanno rigenerati man mano che escono nuovi tornei.

**Popolazione.** I dati provengono da tornei online Limitless, che non coincidono con il
circuito ufficiale dal vivo. La distribuzione ai Mondiali potrebbe differire.

---

## Roadmap

- [ ] **Stima definitiva con NUTS.** L'ADVI sottostima l'incertezza, quindi tutti gli
      intervalli riportati sopra sono ottimistici. Il campionamento Hamiltoniano darà
      intervalli credibili affidabili. Costo elevato: il termine di sinergia rende la
      geometria difficile e il campionamento molto lento, quindi la prima passata è prevista
      senza sinergia.
- [ ] **Confronto formale additivo contro sinergia** sul log-loss held-out, stesso split,
      per stabilire se la sinergia di coppia porta segnale reale o se il metagame è dominato
      dagli effetti principali.
- [ ] **Modello neurale.** Rete siamese con self-attention sui sei Pokémon (Set Transformer):
      la sinergia emerge dall'attenzione invece che da un prodotto scalare a grado 2, e
      cattura interazioni di ordine superiore. Architettura e pipeline già scritte, da
      addestrare e confrontare. Diagramma in [`docs/`](docs/architettura_neural_bt.svg).
      Aspettativa onesta: con 17k match e feature sparse il bayesiano regolarizzato è
      competitivo, e la rete vince solo se l'attenzione cattura struttura che la
      fattorizzazione a grado 2 non vede.
- [ ] **Decadimento temporale** dei pesi per seguire lo spostamento del meta verso agosto.
- [ ] **Estensione della sinergia** a mosse e strumenti, non solo specie.
- [ ] **Aggiornamento continuo** del dataset fino ai Mondiali (28–30 agosto 2026).

---

## Fonti dati

- [Limitless](https://play.limitlesstcg.com) — API pubblica, fonte primaria dei match
- [PokéAPI](https://github.com/PokeAPI/pokeapi) — tabelle nomi multilingua per la canonicalizzazione
- [Pokémon Showdown](https://replay.pokemonshowdown.com) — fonte secondaria opzionale (ladder)

## Licenza

MIT
