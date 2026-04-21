import torch
from torch.utils.data import DataLoader
from dataset import HandwritingDataset, collate_fn
from model import HandwritingSynthesisNetwork, mdn_loss
from vocab import VOCAB_SIZE
import torch.optim as optim
import os
import glob
import re

def main():
    print("Načítám dataset referenčních souborů JSON...")
    dataset = HandwritingDataset("vzorky")
    # S RTX 5060 se toho nemusíme bát, zvednuto na 256 pro maximální vyždímání Tensor Cores:
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    import json
    with open("norm_stats.json", "w") as f:
        json.dump({"mean_x": float(dataset.mean_x), "mean_y": float(dataset.mean_y), "std_x": float(dataset.std_x), "std_y": float(dataset.std_y)}, f)
    print("Normalizační konfigurace vytvořena a uložena do norm_stats.json.")
    
    print(f"Dataset nahrán, velikost: {len(dataset)} ukázek | průměr dx: {dataset.mean_x:.2f}, std dx: {dataset.std_x:.2f}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Běžím na hardware: {device}")
    
    model = HandwritingSynthesisNetwork(VOCAB_SIZE)
    model.to(device)
    
    # RTX 5060 - Torch 2.0 kompilace zabere dlouho, nebo kvůli novém čipu zlobí, proto pro klid běhu přeskočíme:
    # if torch.cuda.is_available() and hasattr(torch, "compile"):
    #     try:
    #         print("Provádím build pro RTX optimalizace (torch.compile)...")
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print("Kompilace se nezdařila, pokračuji standardně bez ní.")
    print("RTX 5060 načteno: Běžím ve zpětně kompatibilním režimu bez hard kompilace (kvůli sm_120 varování).")
            
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else None
    
    start_epoch = 0
    epochs = 1000 # Maximum štědře navýšeno na 1000 epoch, ať to můžeš nechat trénovat klidně celou noc.
    
    # Automatické navázání na předchozí checkpoint
    checkpoint_files = glob.glob("handwriting_model_epoch_*.pt")
    if checkpoint_files:
        epochs_saved = [int(re.findall(r'\d+', f)[-1]) for f in checkpoint_files]
        if epochs_saved:
            latest_epoch = max(epochs_saved)
            checkpoint_path = f"handwriting_model_epoch_{latest_epoch}.pt"
            
            print(f"==================================================")
            print(f"** Našel jsem uloženou síť! Navazuji epochou {latest_epoch + 1} ({checkpoint_path}) **")
            print(f"==================================================")
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                start_epoch = latest_epoch
            except Exception as e:
                print(f"Při otevírání stavu nastala chyba: {e}. Trénink přesto běží bez něj od znova.")
                
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (tokens, strokes, mask) in enumerate(dataloader):
            tokens = tokens.to(device)
            strokes = strokes.to(device)
            mask = mask.to(device)
            
            x_in = strokes[:, :-1, :]
            target_out = strokes[:, 1:, :] 
            target_dx = target_out[:, :, 0]
            target_dy = target_out[:, :, 1]
            target_eos = target_out[:, :, 2]
            target_mask = mask[:, 1:]
            
            optimizer.zero_grad()
            
            # Autocast do Mixed Precision pro RTX grafiky => záchrana VRAM a rychlejší tensor cory.
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", enabled=True):
                    eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho = model(x_in, tokens)
                    loss = mdn_loss(eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_dx, target_dy, target_eos, target_mask)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho = model(x_in, tokens)
                loss = mdn_loss(eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_dx, target_dy, target_eos, target_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epocha {epoch+1}/{epochs} | Dávka {batch_idx}/{len(dataloader)} | Ztráta (Loss): {loss.item():.4f}")
                
        print(f"--- Epocha {epoch+1} byla dokončena s průměrnou ztrátou: {total_loss/len(dataloader):.4f} ---")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"handwriting_model_epoch_{epoch+1}.pt")
            print(f"Stav modelu uložen.")

if __name__ == "__main__":
    main()
