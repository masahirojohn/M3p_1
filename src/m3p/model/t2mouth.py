import torch, torch.nn as nn

class TinyEncoder(nn.Module):
    def __init__(self, vocab, emb_dim=128, n_layers=2, n_heads=4, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.att = nn.Linear(emb_dim,1)
    def forward(self,x,mask):
        h=self.emb(x); h=self.enc(h, src_key_padding_mask=~mask)
        s=self.att(h).squeeze(-1); s=s.masked_fill(~mask, -1e9)
        w=torch.softmax(s,dim=-1).unsqueeze(-1)
        return (h*w).sum(1)  # [B, emb_dim]

class TinyDecoder(nn.Module):
    def __init__(self, emb_dim=128, ncls=6):
        super().__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.head= nn.Linear(emb_dim, ncls)
    def forward(self,c,Tlist):
        maxT=max(Tlist)
        inp = torch.tanh(self.fc(c)).unsqueeze(1).repeat(1,maxT,1)
        out,_ = self.gru(inp)
        return self.head(out)

class Text2Mouth(nn.Module):
    def __init__(self, vocab, emb_dim=128, n_layers=2, n_heads=4, ffn_dim=256, dropout=0.1, ncls=6):
        super().__init__()
        self.enc = TinyEncoder(vocab, emb_dim, n_layers, n_heads, ffn_dim, dropout)
        self.dec = TinyDecoder(emb_dim, ncls)
    def forward(self,x,mask,Tlist):
        c=self.enc(x,mask)
        return self.dec(c,Tlist)
