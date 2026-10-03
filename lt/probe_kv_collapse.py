"""Frozen-weight interventions and block-level measurements for KV collapse.

Uses the same puzzles for every intervention and leaves the training job alone.
The counterfactuals are diagnostic, not evidence of successful retraining.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from . import train as t
from .research_kv_collapse import scalar
from .kv_stability import install


def rms(x):
    return x.float().square().mean().sqrt().detach()


def json_tree(x):
    if isinstance(x,torch.Tensor):
        return scalar(x)
    if isinstance(x,dict):
        return {k:json_tree(v) for k,v in x.items()}
    if isinstance(x,list):
        return [json_tree(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", choices=["test", "saved"], default="test")
    ap.add_argument("--segments", type=int, help="Number of frozen inference segments to probe")
    ap.add_argument("--variants", nargs="+", default=["original", "qk_l2", "read_scaled", "memory_bounded", "pre_ffn_phi", "fp32_kv"])
    args = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    install(ck["cfg"].get("research_variant", "original"))
    cfg = dict(ck["cfg"], batch_size=args.batch, seq_len=81,
               num_puzzle_identifiers=1, activation_checkpoint=False, compile=False,
               nograd_blocks=t.nograd_at(ck["step"], ck["cfg"]))
    if args.data == "saved":
        batch = {k:v[:args.batch].to(args.device) for k,v in ck["rank_states"][0]["carry"]["current_data"].items()}
    else:
        with np.load(cfg["data_npz"]) as data:
            x, y = data["test_inputs"][:args.batch], data["test_labels"][:args.batch]
        batch = dict(inputs=torch.tensor(x.reshape(-1,81).astype(np.int32)+1,device=args.device),
                     labels=torch.tensor(y.reshape(-1,81).astype(np.int32)+1,device=args.device),
                     puzzle_identifiers=torch.zeros(args.batch,dtype=torch.int32,device=args.device))
    recurrent_memory = ck["cfg"].get("research_variant") not in ("current_only","complex_current_only")
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck["step"], data=args.data,
                  recurrent_memory=recurrent_memory, variants={})
    for variant in args.variants:
        torch.manual_seed(0)
        local = dict(cfg)
        if variant == "qk_l2":
            local.update(kv_qk_l2norm=True)
        if variant == "qk_rms":
            local.update(kv_qk_rmsnorm=True)
        if variant == "fp32_kv":
            local.update(kv_projection_fp32=True)
        with torch.device(args.device):
            model = t.LT(local)
        model.load_state_dict({k.removeprefix("model."):v for k,v in ck["raw_model_state_dict"].items()})
        model.eval()
        inner = model.inner
        original_memory = inner.memory_step
        original_boundary = inner.boundary
        original_block = inner.block
        original_phi = inner.phi
        records = []
        current = {}
        hidden_history = []
        prediction_history = []

        def memory(L,q,k,v,*pos,**kw):
            read,m,ek,ev = original_memory(L,q,k,v,*pos,**kw)
            if hasattr(L,'read_lam_raw'):
                with torch.autocast(device_type=args.device,enabled=False):
                    qf,kf,vf=q.float(),k.float(),v.float()
                    qr,kr=inner.apply_rope(qf,L),inner.apply_rope(kf,L)
                    now_matrix=vf.transpose(-1,-2)@kr
                    if inner.config.kv_write_reduction=='mean':
                        now_matrix=now_matrix/k.shape[-2]
                    now=qr@now_matrix.transpose(-1,-2)
                    accumulated=qr@m.transpose(-1,-2)
                    lam=L.read_lam_raw.sigmoid()[None]
                    fast,slow=(1-lam)*now,lam*accumulated
                    current.update(current_kv_read_rms=rms(now),accumulated_read_rms=rms(accumulated),
                        weighted_current_read_rms=rms(fast),weighted_memory_read_rms=rms(slow),
                        read_mix_reconstruction_error=rms(read-fast-slow),
                        read_path_cosine=torch.nn.functional.cosine_similarity(
                            fast.flatten(2),slow.flatten(2),dim=-1).mean().detach())
            old_m = pos[0] if pos else kw.get("memory")
            previous = torch.zeros_like(m) if old_m is None else old_m
            fresh = pos[3] if len(pos)>3 else kw.get("fresh")
            if fresh is not None:
                previous = torch.where(fresh.view(-1,1,1,1),torch.zeros_like(previous),previous)
            current["memory_increment_rms" if recurrent_memory else "current_kv_change_rms"] = rms(m-previous)
            if variant == "read_scaled":
                read = read / inner.dh**0.5
            if variant == "read_operator_bound":
                # Bound the effective read operator without modifying stored M.
                scale = torch.sqrt(1+m.square().sum((-1,-2),keepdim=True)/(16*inner.dh))
                read = read / scale
            if variant == "memory_bounded":
                # A smooth head-wise Frobenius bound, preserving matrix direction.
                scale = torch.sqrt(1+m.square().sum((-1,-2),keepdim=True)/inner.dh)
                m = m / scale
                with torch.autocast(device_type=args.device,enabled=False):
                    read = inner.apply_rope(q.float(),L) @ m.transpose(-1,-2)
            current.update(q_rms=rms(q), k_rms=rms(k), v_rms=rms(v),read_rms=rms(read))
            current["memory_rms" if recurrent_memory else "current_kv_rms"]=rms(m)
            if ek is not None:
                current["trace_k_rms"]=rms(ek)
            if ev is not None:
                current["trace_v_rms"]=rms(ev)
            j = len(records)+1
            if j in (1,2,4,8,16,32,64,128,256):
                s = torch.linalg.svdvals(m.float())
                prefix="memory" if recurrent_memory else "current_kv"
                current[f"{prefix}_sigma_max"]=scalar(s[...,0].max())
                current[f"{prefix}_effective_rank"]=scalar((s.square().sum(-1)/s[...,0].square().clamp_min(1e-30)).mean())
            return read,m,ek,ev

        def boundary(L,h):
            current["pre_ffn_rms"] = rms(h)
            if variant == "pre_ffn_phi":
                h = original_phi(h)
            value = original_boundary(L,h)
            current["ffn_delta_rms"] = rms(value-h)
            current["post_ffn_rms"] = rms(value)
            return value

        def block(*pos,**kw):
            current.clear()
            old_h = pos[1]
            h,m,ek,ev = original_block(*pos,**kw)
            current.update(block=len(records)+1,hidden_rms=rms(h),
                           hidden_change_rms=rms(h-old_h))
            if not hidden_history:
                hidden_history.append(old_h)
            if len(hidden_history)>=2:
                current['hidden_two_step_change_rms']=rms(h-hidden_history[-2])
            hidden_history.append(h)
            del hidden_history[:-2]
            pred=inner.w_cls(h).argmax(-1)
            current['block_accuracy']=(pred==batch['labels']).float().mean()
            if prediction_history:
                current['prediction_flip_rate']=(pred!=prediction_history[-1]).float().mean()
            if len(prediction_history)>=2:
                current['prediction_two_step_flip_rate']=(pred!=prediction_history[-2]).float().mean()
            prediction_history.append(pred)
            del prediction_history[:-2]
            hn = torch.nn.functional.normalize(h.float(),dim=-1)
            current["token_coherence"] = scalar(hn.mean(1).square().sum(-1).mean())
            records.append(dict(current))
            return h,m,ek,ev

        inner.memory_step,inner.boundary,inner.block = memory,boundary,block
        segments=[]
        with torch.no_grad():
            carry=model.initial_carry(batch)
            for segment in range(1,(args.segments or cfg["loops"])+1):
                carry,output=model(carry,batch)
                logits=output["logits"]
                loss=t.stablemax_cross_entropy(logits,batch["labels"]).mean()
                pred=logits.argmax(-1)
                clue=batch["inputs"]!=1
                correct=pred==batch["labels"]
                segments.append(dict(segment=segment,loss=scalar(loss),
                                     accuracy=scalar(correct.float().mean()),
                                     clue_accuracy=scalar(correct[clue].float().mean()),
                                     blank_accuracy=scalar(correct[~clue].float().mean()),
                                     exact=scalar(correct.all(-1).float().mean())))
        result["variants"][variant]=json_tree(dict(segments=segments,blocks=records))
        print(variant,segments[-1],flush=True)
        del model,inner,carry
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
