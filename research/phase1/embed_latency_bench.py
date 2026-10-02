# Minimal BERT/XLM-R encoder in plain torch (2.2.2) loading cached safetensors, to measure CPU latency
# without transformers (global env has transformers 5.9 which refuses torch<2.4). Synthetic text only.
import glob, json, math, os, statistics, sys, time
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

def snap(repo):
    return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]

class Enc:
    def __init__(self, repo):
        d = snap(repo); self.cfg = json.load(open(f"{d}/config.json"))
        w = load_file(f"{d}/model.safetensors"); self.w = {k.replace("bert.", "").replace("roberta.", ""): v.float() for k, v in w.items()}
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json"); self.tok.enable_truncation(512)
        self.xlmr = self.cfg["model_type"] == "xlm-roberta"
        self.nparams = sum(v.numel() for v in w.values()) / 1e6
    def ln(self, x, p): return F.layer_norm(x, (x.shape[-1],), self.w[p + ".weight"], self.w[p + ".bias"], self.cfg["layer_norm_eps"])
    def lin(self, x, p): return F.linear(x, self.w[p + ".weight"], self.w[p + ".bias"])
    @torch.inference_mode()
    def encode(self, texts):
        self.tok.enable_padding(pad_id=self.cfg.get("pad_token_id", 0))
        enc = self.tok.encode_batch(texts)
        ids = torch.tensor([e.ids for e in enc]); mask = torch.tensor([e.attention_mask for e in enc])
        pos = (torch.cumsum(mask, 1) * mask + self.cfg["pad_token_id"]) if self.xlmr else torch.arange(ids.shape[1]).expand_as(ids)
        x = self.w["embeddings.word_embeddings.weight"][ids] + self.w["embeddings.position_embeddings.weight"][pos] \
            + self.w["embeddings.token_type_embeddings.weight"][torch.zeros_like(ids)]
        x = self.ln(x, "embeddings.LayerNorm"); H = self.cfg["num_attention_heads"]; B, T, D = x.shape
        am = (1.0 - mask[:, None, None, :].float()) * -1e9
        for i in range(self.cfg["num_hidden_layers"]):
            p = f"encoder.layer.{i}"
            q, k, v = (self.lin(x, f"{p}.attention.self.{n}").view(B, T, H, D // H).transpose(1, 2) for n in ("query", "key", "value"))
            a = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // H) + am, -1) @ v
            x = self.ln(x + self.lin(a.transpose(1, 2).reshape(B, T, D), f"{p}.attention.output.dense"), f"{p}.attention.output.LayerNorm")
            x = self.ln(x + self.lin(F.gelu(self.lin(x, f"{p}.intermediate.dense")), f"{p}.output.dense"), f"{p}.output.LayerNorm")
        e = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return F.normalize(e, dim=-1)

threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads(); torch.set_num_threads(threads)
print(f"torch {torch.__version__} threads={torch.get_num_threads()}")
passage = ("This section describes the general procedure that applies to bookings made by employees. "
           "Requests must be submitted through the standard process and approved before the event date. ") * 4
for repo, qp, pp in [("sentence-transformers/all-MiniLM-L6-v2", "", ""), ("intfloat/multilingual-e5-base", "query: ", "passage: ")]:
    t0 = time.perf_counter(); m = Enc(repo); load = time.perf_counter() - t0
    # sanity: paraphrase should score above unrelated
    s = m.encode([qp + "How do I cancel a venue booking?", pp + "Venue reservations can be cancelled up to two days before the event.", pp + "The cafeteria serves vegetarian lunch on Fridays."])
    sanity = (float(s[0] @ s[1]), float(s[0] @ s[2]))
    q = qp + "cancellation policy for workshop venues in Pune"; m.encode([q])
    lat = []
    for _ in range(30):
        t = time.perf_counter(); m.encode([q]); lat.append((time.perf_counter() - t) * 1000)
    ntok = len(m.tok.encode(pp + passage).ids)
    t = time.perf_counter()
    for i in range(0, 64, 16): m.encode([pp + passage] * 16)
    bt = time.perf_counter() - t
    print(f"{repo}: params={m.nparams:.0f}M layers={m.cfg['num_hidden_layers']} hidden={m.cfg['hidden_size']} load={load:.2f}s "
          f"sanity(paraphrase={sanity[0]:.3f} > unrelated={sanity[1]:.3f}) query_p50={statistics.median(lat):.1f}ms "
          f"query_p95={sorted(lat)[28]:.1f}ms passages/s@{ntok}tok={64/bt:.1f}")
