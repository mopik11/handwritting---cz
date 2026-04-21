import torch
import numpy as np
import json
import argparse
from model import HandwritingSynthesisNetwork
from vocab import encode_text, VOCAB_SIZE
import matplotlib.pyplot as plt

def sample_gmm(pi, mu_x, mu_y, sigma_x, sigma_y, rho, bias=0.0):
    k = np.random.choice(len(pi), p=pi)
    mean = [mu_x[k], mu_y[k]]
    
    s_x = sigma_x[k] * np.exp(-bias)
    s_y = sigma_y[k] * np.exp(-bias)
    
    cov = [[s_x**2, rho[k]*s_x*s_y],
           [rho[k]*s_x*s_y, s_y**2]]
    
    sample = np.random.multivariate_normal(mean, cov)
    return sample[0], sample[1]

def plot_strokes(points, filename, text=""):
    strokes = []
    current_stroke = []
    for pt in points:
        current_stroke.append((pt[0], pt[1]))
        if pt[2] == 1:
            strokes.append(current_stroke)
            current_stroke = []
    if current_stroke:
        strokes.append(current_stroke)
        
    plt.figure(figsize=(len(text) * 2, 3)) # Dynamická šířka plátna podle délky slova
    for stroke in strokes:
        if len(stroke) > 1:
            xs, ys = zip(*stroke)
            ys = [-y for y in ys] 
            plt.plot(xs, ys, "black", linewidth=2)
            
    plt.axis("equal") # Zabránit roztahování krátkých slov přes celou obrazovku!
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    print(f"Obrázek rukopisu uložen do: {filename}")

def generate_handwriting(text, model_path="handwriting_model_epoch_40.pt", norm_stats_path="norm_stats.json", output="vystup.png", bias=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HandwritingSynthesisNetwork(VOCAB_SIZE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        # V případě, že model byl zkompilován a prefixován, provedeme trim
        state = torch.load(model_path, map_location=device)
        state_dict_new = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state_dict_new)
        
    model.to(device)
    model.eval()
    
    with open(norm_stats_path, "r") as f:
        stats = json.load(f)
    mean_x, mean_y = stats["mean_x"], stats["mean_y"]
    std_x, std_y = stats["std_x"], stats["std_y"]
    
    words = text.split(" ")
    all_generated_points = []
    global_x_offset = 0.0
    
    for word_idx, word in enumerate(words):
        if not word: continue
        
        tokens = torch.tensor(encode_text(word), dtype=torch.long).unsqueeze(0).to(device)
        
        B = 1
        h1 = torch.zeros(B, model.hidden_size, device=device)
        c1 = torch.zeros(B, model.hidden_size, device=device)
        h2 = torch.zeros(B, model.hidden_size, device=device)
        c2 = torch.zeros(B, model.hidden_size, device=device)
        h3 = torch.zeros(B, model.hidden_size, device=device)
        c3 = torch.zeros(B, model.hidden_size, device=device)
        
        text_encoded = model.char_embed(tokens)
        
        x_t = torch.zeros(1, 3, device=device)
        x_t[0, 2] = 1.0 # Předchozí tah skončil (pero je zvednuto), ukotvení kontextu sítě k realitě datasetu.
        w = torch.zeros(1, text_encoded.size(2), device=device)
        kappa = torch.zeros(1, model.window_gaussians, device=device)
        
        generated_points = [[0, 0, 1]] 
        abs_x, abs_y = 0.0, 0.0
        phi_values = []
        
        max_steps = min(3000, tokens.size(1) * 250) # Zvýšený limit tahů, aby síť písmenka dočrtla.
        with torch.no_grad():
            for t in range(max_steps):
                lstm1_in = torch.cat([x_t, w], dim=-1)
                h1, c1 = model.lstm1(lstm1_in, (h1, c1))
                
                w, kappa, phi = model.window(h1, kappa, text_encoded)
                
                lstm2_in = torch.cat([x_t, h1, w], dim=-1)
                h2, c2 = model.lstm2(lstm2_in, (h2, c2))
                
                lstm3_in = torch.cat([x_t, h2, w], dim=-1)
                h3, c3 = model.lstm3(lstm3_in, (h3, c3))
                
                mdn_in = torch.cat([h1, h2, h3], dim=-1)
                eos_hat, pi, mu_x, mu_y, sigma_x, sigma_y, rho = model.mdn(mdn_in)
                
                pi_val = pi.cpu().numpy()[0]
                eos_val = torch.sigmoid(eos_hat).cpu().numpy()[0, 0]
                
                phi_val = phi.cpu().numpy()[0]
                phi_values.append(phi_val)
                # U konce věty se propíšeme s maxem attention dozadu a pak zastavíme když se popíše mezera
                idx_max = np.argmax(phi_val)
                if idx_max == tokens.size(1) - 1 and eos_val > 0.6:
                    break
                    
                dx_norm, dy_norm = sample_gmm(
                    pi_val, mu_x.cpu().numpy()[0], mu_y.cpu().numpy()[0], 
                    sigma_x.cpu().numpy()[0], sigma_y.cpu().numpy()[0], rho.cpu().numpy()[0],
                    bias=bias
                )
                
                dx = dx_norm * std_x + mean_x
                dy = dy_norm * std_y + mean_y
                eos = 1.0 if eos_val > 0.1 else 0.0 # Deterministický threshold zabrání síti drze přeletět celou stránku plnou uhlově rovnou čarou.
                
                abs_x += dx
                abs_y += dy
                
                generated_points.append([abs_x, abs_y, eos])
                
                x_t[0, 0] = dx_norm
                x_t[0, 1] = dy_norm
                x_t[0, 2] = eos

        # Posun aktuálního slova doprava a napojení do hlavního plátna
        for pt in generated_points:
            pt[0] += global_x_offset
            all_generated_points.append(pt)
            
        global_x_offset += abs_x + 50.0 # Mezera mezi slovy
        # Zdvihnout pero mezi slovy
        all_generated_points.append([global_x_offset, 0.0, 1.0])

    # Vizualizace matrice pozornosti posledního slova
    if phi_values:
        phi_matrix = np.array(phi_values).T
        np.save('phi_matrix.npy', phi_matrix)
        plt.figure(figsize=(10, 5))
        plt.imshow(phi_matrix, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Pravděpodobnost pozornosti')
        plt.ylabel('Písmena')
        plt.xlabel('Časové kroky tahu')
        plt.title('Attention (Phi) Během Generování')
        plt.savefig('attention_heatmap.png', bbox_inches='tight')
        plt.close()
    
    plot_strokes(all_generated_points, output, text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generátor ručního písma na základě modelu")
    parser.add_argument("--text", type=str, default="ahoj svete", help="Jaký text má být ručně napsán?")
    parser.add_argument("--model", type=str, default="handwriting_model_epoch_40.pt", help="Cesta k nahranému .pt souboru")
    parser.add_argument("--out", type=str, default="vystup.png", help="Výstupní obrázek .png")
    parser.add_argument("--bias", type=float, default=0.1, help="GMM varianční bias (0.0 = naturální křivky, 1.0+ = mrtvě ustálené rohožky)")
    args = parser.parse_args()
    
    generate_handwriting(args.text, args.model, output=args.out, bias=args.bias)
