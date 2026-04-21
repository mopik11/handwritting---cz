import torch
import json
import os
import glob
from torch.utils.data import Dataset
from vocab import encode_text

class HandwritingDataset(Dataset):
    def __init__(self, data_dir="vzorky"):
        self.files = glob.glob(os.path.join(data_dir, "*.json"))
        self.data = []
        for f in self.files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    d = json.load(file)
                    word = d.get("word") or d.get("text") or "UNKNOWN"
                    pts = d["data"]
                    if len(pts) > 1:
                        self.data.append((word, pts))
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        all_dx = []
        all_dy = []
        for _, pts in self.data:
            for i in range(1, len(pts)):
                all_dx.append(pts[i][0] - pts[i-1][0])
                all_dy.append(pts[i][1] - pts[i-1][1])
                
        if all_dx and all_dy:
            dx_t = torch.tensor(all_dx)
            dy_t = torch.tensor(all_dy)
            self.mean_x, self.std_x = dx_t.mean().item(), dx_t.std().item()
            self.mean_y, self.std_y = dy_t.mean().item(), dy_t.std().item()
        else:
            self.mean_x, self.std_x = 0.0, 1.0
            self.mean_y, self.std_y = 0.0, 1.0
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        word, pts = self.data[idx]
        tokens = encode_text(word)
        
        seq = [[0.0, 0.0, 1.0]] # `<START>` prvek - pero je zdvihnuté před začátkem
        for i in range(1, len(pts)):
            dx = (pts[i][0] - pts[i-1][0] - self.mean_x) / (self.std_x + 1e-8)
            dy = (pts[i][1] - pts[i-1][1] - self.mean_y) / (self.std_y + 1e-8)
            
            # OPRAVA: Pokud předchozí tah skončil (eos=1), jedná se nyní o vzdušný skok do nové pozice
            eos_vzduchem = 1.0 if pts[i-1][2] == 1 else pts[i][2]
            seq.append([dx, dy, eos_vzduchem])
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(seq, dtype=torch.float32)

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[1].size(0), reverse=True)
    
    tokens = [b[0] for b in batch]
    strokes = [b[1] for b in batch]
    
    token_lens = [len(t) for t in tokens]
    max_token_len = max(token_lens)
    tokens_pad = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        tokens_pad[i, :len(t)] = t
        
    stroke_lens = [s.size(0) for s in strokes]
    max_stroke_len = max(stroke_lens)
    strokes_pad = torch.zeros(len(strokes), max_stroke_len, 3, dtype=torch.float32)
    mask = torch.zeros(len(strokes), max_stroke_len, dtype=torch.float32)
    for i, s in enumerate(strokes):
        strokes_pad[i, :s.size(0), :] = s
        mask[i, :s.size(0)] = 1.0
        
    return tokens_pad, strokes_pad, mask
