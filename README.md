# handwritting---cz
## Generátor ručně psaného písma (Handwriting Synthesis)

Tento projekt implementuje architekturu z paperu **"Generating Sequences With Recurrent Neural Networks" (Graves, 2013)**. 
Řešení kombinuje **LSTM** a **Mixture Density Network (MDN)** vrstvy, a implementuje tzv. *Window Layer (Attention)*, který umožňuje síti při produkování dynamicky generovat křivky tahu pomocí textového vstupu (jako podkladu).

Kód byl specificky optimalizovaný pro chod a trénink na paměti a Tensor Core architektuře GPU rodinových karet NVidia, cíleno na specifikaci a schopnosti **RTX 5060** za využití nativního `torch.compile` a smíšené přesnosti vrstev v PyTorch.

## Datová sada
Skript analyzuje složku `vzorky/`, vyhledává JSON soubory, načte vstup a vypočte `dx, dy` (odchylky tahu), a posléze je normalizuje, aby MDN modul nedostával odpaly. Následně z extrahovaných klíčových prvků vytvoří tokenový znakový slovník.

## Požadavky pro běh přes RTX 5060

Nainstalujte požadované knihovny. Optimalizace pro řady s novým CUDou potřebují PyTorch >= 2.0:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy
```

## Soubory kódobáze
- `model.py` - Deklaruje Window Layer, MDN Output Vrstvu a celou obalující `HandwritingSynthesisNetwork`.
- `dataset.py` - Stará se o dataloader, normalizaci přes osu x a y, a zero padding na velikost tensoru v dávce s maskou.
- `train.py` - Samotný učící smyčkový skript stavěn na `amp.autocast` přes cuda optimalizace. Gradient Clipping zastaví divergence.
- `vocab.py` - Tokenizační mapper všech symbolů, slov a znakových bloků potřebných k vyjádření generací.
- `generate.py` - CLI utilitka k vizualizování a vypsání reálných cest tahů pro vstupní "a" až "ž".

## Trénink Modelu
Začněte klasicky:
```bash
python train.py
```
Model se bude ukládat každých 10 epoch pod jménem např. `handwriting_model_epoch_40.pt`. Očekává se, že prvních zhruba 5-10 epoch se loss bude strmě lámat přes generaci průměrových bodů a pak začne window layer (attention mapy) přitahovat k sobě i textová data. Taktéž uloží kalibrační JSON soubor `norm_stats.json` pro de-normalizaci.

## Generování Textů
Model lze po doučení vyzvat ke generování textu:
```bash
python generate.py --text "ahoj svete" --model handwriting_model_epoch_40.pt --out moje_dilo.png
```
Skript přes gmm bias zjemní rušivost křivek a navíc provede i zobrazení např. do obálkových komponent a uloží to pro vás.
