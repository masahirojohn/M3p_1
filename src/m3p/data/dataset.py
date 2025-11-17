import json, torch, torch.utils.data as D

KANA_CHARS = list("ぁあぃいぅうぇえぉおかがきぎくぐけげこご"
                  "さざしじすずせぜそぞただちぢっつづてでとど"
                  "なにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
                  "まみむめもやゃゆゅよょらりるれろわゐゑをんー、。！？.!?\n 　")
C2I = {c:i+1 for i,c in enumerate(KANA_CHARS)}
VOCAB = len(C2I)+1

def text_to_ids(s, max_len=160):
    ids=[C2I.get(c,0) for c in s[:max_len]]; return ids, len(ids)

class MouthDataset(D.Dataset):
    def __init__(self, path):
        self.S=[]
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                o=json.loads(line); self.S.append(o)
    def __len__(self): return len(self.S)
    def __getitem__(self,i):
        o=self.S[i]; x_ids,n = text_to_ids(o["kana"])
        return {"x":torch.tensor(x_ids).long(), "n":n, "T":o["T"], "y":torch.tensor(o["mouth_steps"]).long()}

def collate(B):
    maxL=max(b["n"] for b in B); maxT=max(b["T"] for b in B); N=len(B)
    x=torch.zeros(N,maxL).long(); mask=torch.zeros(N,maxL).bool()
    y=torch.full((N,maxT),-100).long()
    for i,b in enumerate(B):
        x[i,:b["n"]]=b["x"]; mask[i,:b["n"]]=1; y[i,:b["T"]]=b["y"]
    return {"x":x,"mask":mask,"y":y,"T_list":[b["T"] for b in B]}
