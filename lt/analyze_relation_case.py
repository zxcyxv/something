"""Exact edge removals and continued inference for one illustrative correction.

The event and ranked edges are selected retrospectively using normal outcomes.
This is a causal diagnostic of this case, not a label-free inference algorithm.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
from analyze_relation_transport import RelationBlocks
from ckpt_npz import load_data, load_lt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/relation_transport_v11")
    ap.add_argument("--event", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.root)
    report = json.loads((root/"summary.json").read_text())
    ev = report["events"]["events"][args.event]
    normal_pred = np.load(root/"normal.npz")["pred"]
    edge = np.load(root/"event_edges.npz")
    row = int(np.flatnonzero(edge["event"] == args.event)[0])
    ranking = np.argsort(edge["contribution"][row].ravel())[::-1]
    top = [tuple(map(int,np.unravel_index(i, edge["contribution"][row].shape))) for i in ranking[:6]]
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    n = normal_pred.shape[1]
    model, _, _ = load_lt("checkpoints/v1.1_step160000.npz",mod=train,batch_size=n,loops=129,amp=False)
    _, _, batch = load_data(n=n); b = RelationBlocks(model,batch)
    h = b.inner.init_hidden.expand(n,81,-1).clone(); state = None
    for k in range(1,ev["block"]):
        p = b.parts(h,state); h=b.read_parts(p);state=p["state"]
    snapshot=(h,state)
    puzzle,cell=ev["puzzle"],ev["cell"]
    gold,old=ev["gold"],ev["previous_choice"]
    p=b.parts(h,state);hn,pre=b.read_parts(p,return_pre=True)
    torch.testing.assert_close(b.inner.w_cls(hn).argmax(-1).cpu(),torch.from_numpy(normal_pred[ev["block"]]).long(),atol=0,rtol=0)
    def score(hcell):
        logits=b.inner.w_cls(hcell)
        competitor=logits.clone();competitor[gold]=-torch.inf
        return {"prediction_digit":int(logits.argmax())-1,
                "gold_margin":float(logits[gold]-competitor.max()),
                "gold_logit":float(logits[gold]),"previous_choice_logit":float(logits[old])}
    results={"event":ev,"human_target_rc":[cell//9+1,cell%9+1],
             "normal":score(hn[puzzle,cell]),"single_edges":[],"continuations":{}}
    for head,source in top:
        coefficient=b.lam[head,0,0]*p["symmetric"][puzzle,head,cell,source]
        delta=coefficient*(p["v"][puzzle,source,head]@b.layer.w_sh[head])
        cf=b.inner.phi(pre[puzzle,cell]-delta)
        fields={key:float(edge[key][row,head,source]) for key in
                ("contribution","gold_logit_contribution","previous_logit_contribution",
                 "memory","current_write","phase","phase_mean","agree","agree_mean","source_direction_cosine")}
        results["single_edges"].append(dict(head=head,source=source,source_rc=[source//9+1,source%9+1],
                                           source_digit=int(normal_pred[ev["block"]-1,puzzle,source])-1,
                                           source_correct=bool(edge["source_correct"][row,source]),
                                           source_clue=bool(edge["source_clue"][row,source]),
                                           removed=score(cf),**fields))
    # Exact algebraic split of the normal output margin, using its common phi
    # denominator. This is not a causal attribution of that denominator.
    decoder=b.inner.w_cls.weight[gold]-b.inner.w_cls.weight[old]
    denom=(1+pre[puzzle,cell].square().sum()/b.inner.d).sqrt()
    coeffs={"instant_read":(1-b.lam)*p["a"],"instant_write":b.lam*p["target"],
            "history":b.lam*(p["w"]-p["target"])}
    margin_parts={"prepared_hidden":float((decoder*p["q"][puzzle,cell]).sum()/denom),
                  "classifier_bias":float(b.inner.w_cls.bias[gold]-b.inner.w_cls.bias[old])}
    for name,c in coeffs.items():
        message=torch.einsum("hn,nhc,hcd->d",c[puzzle,:,cell],p["v"][puzzle],b.layer.w_sh)
        margin_parts[name]=float((decoder*message).sum()/denom)
    results["normal_fixed_margin_algebraic_split"]=margin_parts
    results["normal_fixed_margin_split_sum"]=sum(margin_parts.values())
    masks={}
    for count in (1,3):
        mask=torch.zeros((b.inner.H,81),device="cuda")
        for head,source in top[:count]:mask[head,source]=1
        masks[count]=mask
    modes=("pulse_top1","persistent_top1","persistent_top3","pulse_all_agree_target","persistent_all_agree_target")
    labels=batch["labels"][puzzle].cpu().numpy()
    def describe(pred):
        correct=(pred==labels).all(-1);target=pred[:,cell]==labels[cell]
        return {"first_puzzle_correct_block":int(np.flatnonzero(correct)[0]+ev["block"]) if correct.any() else None,
                "first_target_correct_block":int(np.flatnonzero(target)[0]+ev["block"]) if target.any() else None,
                "final_puzzle_correct":bool(correct[-1]),"final_target_correct":bool(target[-1]),
                "final_error_count":int((pred[-1]!=labels).sum())}
    trajectories={"normal":normal_pred[ev["block"]:,puzzle]}
    results["continuations"]["normal"]=describe(trajectories["normal"])
    for mode in modes:
        h,state=snapshot
        predictions=[]
        for k in range(ev["block"],len(normal_pred)):
            p=b.parts(h,state);hn,pre=b.read_parts(p,return_pre=True)
            active=mode.startswith("persistent") or k==ev["block"]
            if active:
                if mode.endswith("agree_target"):
                    c=p["agree_history"][puzzle,:,cell]
                else:
                    c=p["symmetric"][puzzle,:,cell]*masks[3 if mode.endswith("top3") else 1]
                delta=torch.einsum("hn,nhc,hcd->d",b.lam[:,0,0,None]*c,p["v"][puzzle],b.layer.w_sh)
                hn=hn.clone();hn[puzzle,cell]=b.inner.phi(pre[puzzle,cell]-delta)
            h,state=hn,p["state"]
            predictions.append(b.inner.w_cls(h[puzzle]).argmax(-1).cpu().numpy().astype(np.int8))
        trajectories[mode]=np.stack(predictions)
        results["continuations"][mode]=describe(trajectories[mode])
        print(mode,results["continuations"][mode],flush=True)
        (root/f"case_{args.event}.json").write_text(json.dumps(results,indent=2))
    np.savez_compressed(root/f"case_{args.event}_trajectories.npz",**trajectories)
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":main()
