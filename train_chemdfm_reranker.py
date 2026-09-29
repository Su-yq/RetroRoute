#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, inspect, json, math, os, random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SCHEMA_VERSION = "chemdfm_route_rerank_v2_v1"

def write_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sint(x, d=0):
    try: return int(x)
    except: return d

def yn(x): return "yes" if bool(x) else "no"

def build_prompt(row, candidates, display_ids, max_history_steps=6,
                 include_molt5_rank=True, include_validity=True, include_stock=True):
    state = row.get("state", {}) or {}
    target = str(row.get("target_smiles", "")).strip()
    current = str(state.get("current_smiles", "")).strip()
    depth = sint(state.get("current_depth", 0))
    max_depth = sint(state.get("max_depth", 14), 14)
    remaining = sint(state.get("remaining_depth_budget", max_depth-depth), max_depth-depth)
    frontier = state.get("frontier", []) or []
    history = state.get("history", []) or []
    if max_history_steps > 0: history = history[-max_history_steps:]

    z = [
        "[Round 0]",
        "Human: You are a retrosynthesis candidate reranker.",
        "",
        "Select the most promising one-step retrosynthetic candidate for the CURRENT molecule, considering its role in the FULL multi-step synthesis.",
        "The objective is to maximize the likelihood of completing the final target from purchasable starting materials within the search budget.",
        "Use the route context when comparing candidates.",
        "Do not propose new reactants. Choose only from the candidates provided.",
        "",
        "[FINAL TARGET]", target, "",
        "[SEARCH STATE]",
        f"Current molecule: {current}",
        f"Current depth: {depth}",
        f"Maximum search depth: {max_depth}",
        f"Remaining depth budget: {remaining}",
        "",
        "Current frontier:"
    ]
    if frontier:
        for i, x in enumerate(frontier, 1):
            z.append(f"  F{i}. {x.get('smiles','')} | depth={sint(x.get('depth',0))} | purchasable={yn(x.get('in_stock',False))}")
    else:
        z.append("  None")

    z += ["", "[ROUTE HISTORY]"]
    if history:
        for h in history:
            rs = str(h.get("reactants_str", "")).strip()
            if not rs:
                rs = ".".join(str(x) for x in (h.get("reactants_smiles", []) or []))
            z.append(f"  Step {sint(h.get('step',0))}: {h.get('product_smiles','')} -> {rs}")
    else:
        z.append("  None")

    z += ["", "[CANDIDATES]"]
    for cid, c in zip(display_ids, candidates):
        z.append(f"{cid}. Reactants: {c.get('reactants_str','')}")
        if include_molt5_rank:
            z.append(f"   MolT5 rank: {sint(c.get('molt5_rank',0))}")
        if include_validity:
            z.append(f"   Structurally valid: {yn(c.get('valid',False))}")
        if include_stock:
            z.append(f"   Purchasable reactants: {sint(c.get('stock_count',0))}/{sint(c.get('num_reactants',0))}")

    z += ["", "[OUTPUT]", "Return only the ID of the preferred candidate.", "Assistant:"]
    return "\n".join(z)

class JsonlDataset(Dataset):
    def __init__(self, path, training, randomize_ids, seed, max_candidates,
                 max_history_steps, include_molt5_rank, include_validity, include_stock,
                 max_samples=-1):
        self.path = Path(path)
        self.training = training
        self.randomize_ids = bool(randomize_ids and training)
        self.seed = seed; self.epoch = 0
        self.max_candidates = max_candidates
        self.max_history_steps = max_history_steps
        self.include_molt5_rank = include_molt5_rank
        self.include_validity = include_validity
        self.include_stock = include_stock
        self.offsets = []
        self.stats = dict(rows_seen=0, rows_kept=0, skip_not_trainable=0,
                          skip_no_candidates=0, skip_no_positive=0, multi_positive=0)
        off = 0
        with open(self.path, "rb") as f:
            while True:
                line = f.readline()
                if not line: break
                self.stats["rows_seen"] += 1
                try: row = json.loads(line.decode("utf-8"))
                except:
                    off = f.tell(); continue
                sup = row.get("supervision", {}) or {}
                if not sup.get("trainable_v2", False):
                    self.stats["skip_not_trainable"] += 1; off = f.tell(); continue
                cands = (row.get("candidates", []) or [])[:max_candidates]
                if not cands:
                    self.stats["skip_no_candidates"] += 1; off = f.tell(); continue
                ids = {str(c.get("candidate_id","")) for c in cands}
                pos = set(str(x) for x in (sup.get("positive_candidate_ids",[]) or [])) & ids
                if not pos:
                    self.stats["skip_no_positive"] += 1; off = f.tell(); continue
                if len(pos) > 1: self.stats["multi_positive"] += 1
                self.offsets.append(off); self.stats["rows_kept"] += 1
                if max_samples > 0 and len(self.offsets) >= max_samples: break
                off = f.tell()
        self._fh = None

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.offsets)

    def _row(self, idx):
        if self._fh is None: self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[idx])
        return json.loads(self._fh.readline().decode("utf-8"))

    def __getitem__(self, idx):
        row = self._row(idx)
        cands = list(row.get("candidates", []) or [])[:self.max_candidates]
        old_pos_ids = set(str(x) for x in row["supervision"]["positive_candidate_ids"])
        order = list(range(len(cands)))
        if self.randomize_ids and len(order) > 1:
            rng = random.Random(self.seed + 1000003*self.epoch + idx)
            rng.shuffle(order)
        shown = [cands[i] for i in order]
        display_ids = LABELS[:len(shown)]
        positives, molt5_ranks = [], []
        for new_pos, old_pos in enumerate(order):
            c = cands[old_pos]
            if str(c.get("candidate_id","")) in old_pos_ids: positives.append(new_pos)
            molt5_ranks.append(sint(c.get("molt5_rank", old_pos+1), old_pos+1))
        prompt = build_prompt(
            row, shown, display_ids, self.max_history_steps,
            self.include_molt5_rank, self.include_validity, self.include_stock
        )
        return dict(
            sample_id=row.get("sample_id"),
            prompt=prompt,
            num_candidates=len(shown),
            positive_positions=positives,
            molt5_ranks=molt5_ranks,
            current_depth=sint((row.get("state",{}) or {}).get("current_depth",0))
        )

def choose_label_tokens(tokenizer, max_candidates):
    base = LABELS[:max_candidates]
    for texts in ([f" {x}" for x in base], base):
        ids=[]; ok=True
        for t in texts:
            e=tokenizer.encode(t, add_special_tokens=False)
            if len(e)!=1: ok=False; break
            ids.append(e[0])
        if ok and len(set(ids))==len(ids): return texts, ids
    raise RuntimeError("A-J are not single-token labels for this tokenizer. Inspect tokenizer.encode(' A') / tokenizer.encode('A').")

def collate_fn(tokenizer, max_length, max_candidates):
    def collate(batch):
        enc = tokenizer([x["prompt"] for x in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        b=len(batch)
        cm=torch.zeros((b,max_candidates),dtype=torch.bool)
        pm=torch.zeros((b,max_candidates),dtype=torch.bool)
        mr=torch.full((b,max_candidates),10**6,dtype=torch.long)
        meta=[]
        for i,x in enumerate(batch):
            n=x["num_candidates"]; cm[i,:n]=True
            for p in x["positive_positions"]: pm[i,p]=True
            for j,r in enumerate(x["molt5_ranks"]): mr[i,j]=r
            meta.append(dict(sample_id=x["sample_id"],current_depth=x["current_depth"]))
        return dict(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"],
                    candidate_mask=cm,positive_mask=pm,molt5_ranks=mr,metadata=meta)
    return collate

def to_device(batch, device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def logits_keep_arg(model):
    objs = [model]
    if hasattr(model, "get_base_model"):
        try:
            objs.append(model.get_base_model())
        except Exception:
            pass
    for obj in objs:
        try:
            p = inspect.signature(obj.forward).parameters
        except Exception:
            continue
        if "logits_to_keep" in p:
            return "logits_to_keep"
        if "num_logits_to_keep" in p:
            return "num_logits_to_keep"
    return None

def forward_scores(model, batch, label_ids, keep_arg):
    kw=dict(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
            use_cache=False,return_dict=True)
    if keep_arg: kw[keep_arg]=1
    out=model(**kw)
    return out.logits[:,-1,:].index_select(-1,label_ids).float()

def listwise_loss(scores, cm, pm):
    neg=torch.finfo(scores.dtype).min
    return (torch.logsumexp(scores.masked_fill(~cm,neg),-1) -
            torch.logsumexp(scores.masked_fill(~pm,neg),-1)).mean()

def rank_stats(scores, cm, pm):
    scores=scores.detach().cpu(); cm=cm.cpu(); pm=pm.cpu()
    hits={1:0,3:0,5:0,10:0}; rr=0.0
    for i in range(scores.shape[0]):
        vi=torch.where(cm[i])[0]
        order=vi[torch.argsort(scores[i,vi],descending=True)].tolist()
        pos=set(torch.where(pm[i])[0].tolist())
        br=next((r for r,j in enumerate(order,1) if j in pos),None)
        if br:
            rr += 1/br
            for k in hits:
                if br<=k: hits[k]+=1
    n=scores.shape[0]
    return {**{f"hit@{k}":hits[k]/n for k in hits},"mrr":rr/n}

def molt5_stats(ranks, cm, pm):
    ranks=ranks.cpu(); cm=cm.cpu(); pm=pm.cpu()
    hits={1:0,3:0,5:0,10:0}; rr=0.0; n=ranks.shape[0]
    for i in range(n):
        pp=torch.where(pm[i]&cm[i])[0].tolist()
        br=min(int(ranks[i,j]) for j in pp)
        rr += 1/br
        for k in hits:
            if br<=k: hits[k]+=1
    return {**{f"molt5_hit@{k}":hits[k]/n for k in hits},"molt5_mrr":rr/n}

@torch.no_grad()
def evaluate(model, loader, device, label_ids, keep_arg, amp, amp_dtype, max_batches=-1):
    model.eval(); n=0; loss_sum=0.0
    agg={k:0.0 for k in ["hit@1","hit@3","hit@5","hit@10","mrr",
                          "molt5_hit@1","molt5_hit@3","molt5_hit@5","molt5_hit@10","molt5_mrr"]}
    depth={}
    for bi,batch in enumerate(tqdm(loader,desc="valid",dynamic_ncols=True)):
        if max_batches>0 and bi>=max_batches: break
        batch=to_device(batch,device); bs=batch["input_ids"].shape[0]
        with torch.autocast("cuda",dtype=amp_dtype,enabled=(amp and device.type=="cuda")):
            s=forward_scores(model,batch,label_ids,keep_arg)
            l=listwise_loss(s,batch["candidate_mask"],batch["positive_mask"])
        n+=bs; loss_sum+=l.item()*bs
        a=rank_stats(s,batch["candidate_mask"],batch["positive_mask"])
        b=molt5_stats(batch["molt5_ranks"],batch["candidate_mask"],batch["positive_mask"])
        for k,v in {**a,**b}.items(): agg[k]+=v*bs
        sc=s.detach().cpu(); cm=batch["candidate_mask"].cpu(); pm=batch["positive_mask"].cpu()
        for i,m in enumerate(batch["metadata"]):
            d=m["current_depth"]; depth.setdefault(d,[0,0,0.0]); depth[d][0]+=1
            vi=torch.where(cm[i])[0]; order=vi[torch.argsort(sc[i,vi],descending=True)].tolist()
            pos=set(torch.where(pm[i])[0].tolist())
            br=next((r for r,j in enumerate(order,1) if j in pos),None)
            if br: depth[d][1]+=int(br==1); depth[d][2]+=1/br
    out=dict(loss=loss_sum/n if n else None,num_rows=n)
    if n:
        out.update({k:v/n for k,v in agg.items()})
    out["by_current_depth"]={str(d):{"rows":x[0],"hit@1":x[1]/x[0],"mrr":x[2]/x[0]} for d,x in sorted(depth.items())}
    return out

def choose_targets(model, requested):
    suffix={n.split(".")[-1] for n,_ in model.named_modules()}
    got=[x for x in requested if x in suffix]
    if not got: raise RuntimeError(f"No LoRA targets found. Requested={requested}")
    return got

def param_stats(model):
    t=sum(p.numel() for p in model.parameters())
    tr=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable=tr,total=t,trainable_percent=100*tr/t)

def save_ckpt(model, tok, path, args, epoch, metrics, label_texts, label_ids):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(path); tok.save_pretrained(path)
    write_json(dict(epoch=epoch,metrics=metrics,args=vars(args),
                    label_texts=label_texts,label_token_ids=label_ids),path/"trainer_state.json")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_jsonl",default="./chemdfm_dataset/train_chemdfm.jsonl")
    ap.add_argument("--valid_jsonl",default="./chemdfm_dataset/valid_chemdfm.jsonl")
    ap.add_argument("--model_dir",default="../ChemDFM-local-model")
    ap.add_argument("--output_dir",default="./chemdfm_reranker_v2")
    ap.add_argument("--max_candidates",type=int,default=10)
    ap.add_argument("--max_history_steps",type=int,default=6)
    ap.add_argument("--max_length",type=int,default=4096)
    ap.add_argument("--include_molt5_rank",action="store_true")
    ap.add_argument("--include_validity",action="store_true")
    ap.add_argument("--include_stock",action="store_true")
    ap.add_argument("--randomize_candidate_ids",action="store_true")
    ap.add_argument("--epochs",type=int,default=2)
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--eval_batch_size",type=int,default=2)
    ap.add_argument("--grad_accum_steps",type=int,default=16)
    ap.add_argument("--learning_rate",type=float,default=1e-4)
    ap.add_argument("--weight_decay",type=float,default=0.01)
    ap.add_argument("--warmup_ratio",type=float,default=0.03)
    ap.add_argument("--max_grad_norm",type=float,default=1.0)
    ap.add_argument("--lora_r",type=int,default=16)
    ap.add_argument("--lora_alpha",type=int,default=32)
    ap.add_argument("--lora_dropout",type=float,default=0.05)
    ap.add_argument("--lora_target_modules",nargs="+",
                    default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ap.add_argument("--load_in_4bit",action="store_true")
    ap.add_argument("--bf16",action="store_true")
    ap.add_argument("--fp16",action="store_true")
    ap.add_argument("--gradient_checkpointing",action="store_true")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--logging_steps",type=int,default=20)
    ap.add_argument("--save_every_epoch",action="store_true")
    ap.add_argument("--early_stop_patience",type=int,default=2)
    ap.add_argument("--max_train_samples",type=int,default=-1)
    ap.add_argument("--max_valid_samples",type=int,default=-1)
    ap.add_argument("--eval_max_batches",type=int,default=-1)
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    if args.bf16 and args.fp16: raise ValueError("Choose only one of --bf16/--fp16")
    set_seed(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); write_json(vars(args),out/"config.json")

    train=JsonlDataset(args.train_jsonl,True,args.randomize_candidate_ids,args.seed,args.max_candidates,
                       args.max_history_steps,args.include_molt5_rank,args.include_validity,args.include_stock,args.max_train_samples)
    valid=JsonlDataset(args.valid_jsonl,False,False,args.seed,args.max_candidates,
                       args.max_history_steps,args.include_molt5_rank,args.include_validity,args.include_stock,args.max_valid_samples)
    print("train:",train.stats); print("valid:",valid.stats)

    tok=AutoTokenizer.from_pretrained(args.model_dir,trust_remote_code=True,use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    tok.padding_side="left"; tok.truncation_side="left"
    label_texts,label_id_list=choose_label_tokens(tok,args.max_candidates)
    print("label tokens:",list(zip(label_texts,label_id_list)))

    dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    qconf=None
    if args.load_in_4bit:
        qconf=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=(torch.bfloat16 if args.bf16 else torch.float16))
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    kw=dict(trust_remote_code=True)
    if qconf is not None:
        kw.update(quantization_config=qconf,device_map={"":0})
    else:
        kw["torch_dtype"]=dtype
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,**kw)
    if not args.load_in_4bit: model.to(device)
    model.config.use_cache=False
    if getattr(model.config,"pad_token_id",None) is None: model.config.pad_token_id=tok.pad_token_id
    if args.load_in_4bit:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing and hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    targets=choose_targets(model,args.lora_target_modules)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=args.lora_r,
                        lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,
                        target_modules=targets,bias="none"))
    print("LoRA targets:",targets); print("params:",param_stats(model))
    write_json(dict(train_scan=train.stats,valid_scan=valid.stats,lora_targets=targets,
                    params=param_stats(model),label_texts=label_texts,label_ids=label_id_list),out/"setup.json")

    coll=collate_fn(tok,args.max_length,args.max_candidates)
    tl=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,collate_fn=coll)
    vl=DataLoader(valid,batch_size=args.eval_batch_size,shuffle=False,num_workers=0,pin_memory=True,collate_fn=coll)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=args.weight_decay)
    updates=math.ceil(len(tl)/max(1,args.grad_accum_steps))*args.epochs
    sch=get_cosine_schedule_with_warmup(opt,int(updates*args.warmup_ratio),max(1,updates))
    label_ids=torch.tensor(label_id_list,dtype=torch.long,device=device)
    keep=logits_keep_arg(model); print("last-logits optimization:",keep)
    amp=(args.bf16 or args.fp16) and device.type=="cuda"; amp_dtype=torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type=="cuda"))

    best=-1.0; best_loss=1e30; stale=0; hist=[]; gstep=0; ustep=0
    opt.zero_grad(set_to_none=True)
    for epoch in range(1,args.epochs+1):
        train.set_epoch(epoch); model.train(); ls=0.0; nr=0
        pbar=tqdm(tl,desc=f"train {epoch}/{args.epochs}",dynamic_ncols=True)
        for step,batch in enumerate(pbar,1):
            gstep+=1; batch=to_device(batch,device); bs=batch["input_ids"].shape[0]
            with torch.autocast("cuda",dtype=amp_dtype,enabled=amp):
                s=forward_scores(model,batch,label_ids,keep)
                loss=listwise_loss(s,batch["candidate_mask"],batch["positive_mask"])
                scaled_loss = loss/max(1,args.grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            ls+=loss.item()*bs; nr+=bs
            if step%args.grad_accum_steps==0 or step==len(tl):
                if args.max_grad_norm>0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.max_grad_norm)
                scaler.step(opt); scaler.update()
                sch.step(); opt.zero_grad(set_to_none=True); ustep+=1
            if gstep%args.logging_steps==0:
                pbar.set_postfix(loss=f"{ls/max(1,nr):.4f}",lr=f"{sch.get_last_lr()[0]:.2e}",updates=ustep)

        vm=evaluate(model,vl,device,label_ids,keep,amp,amp_dtype,args.eval_max_batches)
        rec=dict(epoch=epoch,train_loss=ls/max(1,nr),valid=vm,global_step=gstep,update_step=ustep)
        hist.append(rec); write_json(hist,out/"training_history.json")
        print(json.dumps(rec,ensure_ascii=False,indent=2))
        h1=float(vm.get("hit@1",0)); vlss=float(vm["loss"])
        improved=(h1>best+1e-8) or (abs(h1-best)<=1e-8 and vlss<best_loss)
        if improved:
            best=h1; best_loss=vlss; stale=0
            save_ckpt(model,tok,out/"checkpoint-best",args,epoch,vm,label_texts,label_id_list)
        else:
            stale+=1
        if args.save_every_epoch:
            save_ckpt(model,tok,out/f"checkpoint-epoch-{epoch}",args,epoch,vm,label_texts,label_id_list)
        if args.early_stop_patience>0 and stale>=args.early_stop_patience: break

    write_json(dict(best_valid_hit1=best,best_valid_loss=best_loss,epochs_completed=len(hist),
                    checkpoint_best=str(out/"checkpoint-best")),out/"summary.json")

if __name__=="__main__":
    main()