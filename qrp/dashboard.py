"""Static HTML dashboard: one self-contained file, no server, opens in any browser.

Shows every trial (filterable), candidates with equity and drawdown curves, the family
walk-forward OOS curves, and holdout curves, each labelled with where its numbers come from.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .registry import Registry

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strategy Research Dashboard</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--a:#2f6fb3;--b:#b8752a;--neg:#b33a3a;--pos:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a09f98;--line:#33332f;--card:#1f1f1d;--a:#6ea8e6;--b:#e0a35c;--neg:#e07070;--pos:#6cc391}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:24px 0 8px}
.note{color:var(--mut)}.warn{border-left:3px solid var(--b);padding:6px 10px;background:var(--card)}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border:1px solid var(--line);border-radius:9px;color:var(--mut);margin-left:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(340px,100%),1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h3{font-size:13px;margin:0 0 4px;word-break:break-word}.card div{overflow-wrap:anywhere}svg{width:100%;height:auto;display:block}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
.tablewrap{overflow:auto;max-height:640px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{cursor:pointer;position:sticky;top:0;background:var(--card)}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}.pass{color:var(--pos)}.fail{color:var(--neg)}
.legend span{margin-right:10px}.sw{display:inline-block;width:10px;height:3px;vertical-align:middle;margin-right:4px}
</style></head><body><main>
<h1>Strategy research dashboard</h1><div class="note" id="meta"></div>
<div id="banner"></div>
<h2>Family walk-forward (out-of-sample within dev) <span class="tag">WALK-FORWARD OOS</span></h2>
<div class="grid" id="wf"></div>
<h2>Candidates <span class="tag">IN-SAMPLE (DEV)</span></h2>
<div class="grid" id="cands"></div>
<h2>Final holdout <span class="tag">HOLDOUT</span></h2><div id="holdout" class="grid"></div>
<h2>All configurations tested <span class="tag">IN-SAMPLE (DEV)</span></h2>
<div class="controls">
<select id="f_family"><option value="">all families</option></select>
<select id="f_signal"><option value="">all signals</option></select>
<select id="f_filter"><option value="">all filters</option></select>
<select id="f_exit"><option value="">all exits</option></select>
<select id="f_sizing"><option value="">all sizing</option></select>
<label>max DD ≤ <input id="f_dd" type="number" step="0.05" value="1" style="width:70px"></label>
<label>min trades <input id="f_tr" type="number" value="0" style="width:70px"></label>
<span class="note" id="count"></span></div>
<div class="tablewrap"><table id="t"><thead><tr></tr></thead><tbody></tbody></table></div>
<p class="note">Sharpe ratios are annualised, versus zero (cash earns nothing in the engine). Alpha Sharpe removes what the strategy's average market exposure (beta to the benchmark) would have earned.</p>
</main><script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const fmt = (x, p) => x == null || isNaN(x) ? "n/a" : p ? (x*100).toFixed(1)+"%" : (+x).toFixed(2);
function cum(r){let e=1;return r.map(x=>e*=1+x)}
function dd(r){let e=1,pk=1;return r.map(x=>{e*=1+x;pk=Math.max(pk,e);return e/pk-1})}
function chart(series, h, opts={}){
  const W=600,H=h,pad=4;let lo=Infinity,hi=-Infinity,n=0;
  series.forEach(s=>{s.v.forEach(x=>{if(x<lo)lo=x;if(x>hi)hi=x});n=Math.max(n,s.v.length)});
  if(opts.zero){hi=Math.max(hi,0)} if(hi===lo){hi+=1e-9}
  const y=v=>pad+(H-2*pad)*(1-(v-lo)/(hi-lo)), x=i=>i/(Math.max(n-1,1))*W;
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
  if(opts.base!=null&&opts.base>lo&&opts.base<hi) svg+=`<line x1="0" x2="${W}" y1="${y(opts.base)}" y2="${y(opts.base)}" stroke="var(--line)"/>`;
  series.forEach(s=>{const step=Math.max(1,Math.floor(s.v.length/600));let pts=[];for(let i=0;i<s.v.length;i+=step)pts.push(x(i).toFixed(1)+","+y(s.v[i]).toFixed(1));
    svg+=`<polyline fill="none" stroke="${s.c}" stroke-width="1.5" points="${pts.join(" ")}"/>`});
  return svg+"</svg>";
}
function legend(items){return '<div class="legend note">'+items.map(([c,t])=>`<span><i class="sw" style="background:${c}"></i>${t}</span>`).join("")+"</div>"}
$("#meta").textContent = `Study ${D.study.id} · ${D.study.name} · dev ${D.study.split.dev_start} → ${D.study.split.dev_end} · holdout ${D.study.split.holdout_start} → ${D.study.split.holdout_end} · ${D.n_trials} configs tested on this data`;
if(D.synthetic) $("#banner").innerHTML = '<p class="warn">Simulated data. These results test the software and say nothing about real markets.</p>';
// walk-forward
D.wf.forEach(f=>{const b=D.bench.slice(f.start, f.start+f.returns.length);
  $("#wf").insertAdjacentHTML("beforeend",`<div class="card"><h3>${f.family}</h3><div class="note">OOS Sharpe ${fmt(f.sharpe)} · alpha Sharpe ${fmt(f.alpha_sharpe)} · benchmark ${fmt(f.bench_sharpe)} · PBO ${fmt(f.pbo)} · SPA p ${fmt(f.spa)}</div>`+
  chart([{v:cum(b),c:"var(--b)"},{v:cum(f.returns),c:"var(--a)"}],170,{base:1})+legend([["var(--a)","selection procedure, OOS"],["var(--b)",D.bench_symbol+" buy & hold"]])+"</div>")});
// candidates
D.cands.forEach(c=>{const b=D.bench.slice(D.score_start, D.score_start+c.returns.length);
  $("#cands").insertAdjacentHTML("beforeend",`<div class="card"><h3>${c.family} · <code>${c.hash}</code></h3><div class="note">${c.desc}</div>
  <div>Sharpe ${fmt(c.sharpe)} · alpha ${fmt(c.alpha_sharpe)} · CAGR ${fmt(c.cagr,1)} · maxDD ${fmt(c.mdd,1)} · DSR ${fmt(c.dsr)} · <b class="${c.status.includes("pass")?"pass":"fail"}">${c.status}</b></div>
  <div class="note">${c.failed.length?"failed: "+c.failed.join(", "):"all gates passed"}</div>`+
  chart([{v:cum(b),c:"var(--b)"},{v:cum(c.returns),c:"var(--a)"}],150,{base:1})+chart([{v:dd(c.returns),c:"var(--neg)"}],70,{zero:true,base:0})+
  legend([["var(--a)","strategy"],["var(--b)",D.bench_symbol],["var(--neg)","drawdown"]])+"</div>")});
if(!D.cands.length) $("#cands").innerHTML='<p class="note">No candidates.</p>';
// holdout
if(!D.holdout){$("#holdout").innerHTML='<p class="note">Not opened yet. It can be opened once, after the candidate list is final.</p>'}
else{ if(D.holdout.contaminated) $("#holdout").insertAdjacentHTML("beforebegin",'<p class="warn">This holdout was opened more than once, so it is no longer a clean test.</p>');
  const hs=Object.entries(D.holdout.strategies); if(!hs.length) $("#holdout").innerHTML='<p class="note">Opened, but no candidate had passed the dev gates.</p>';
  hs.forEach(([h,s])=>{$("#holdout").insertAdjacentHTML("beforeend",`<div class="card"><h3>${s.family} · <code>${h}</code></h3><div class="note">${s.description}</div>
  <div>Holdout Sharpe ${fmt(s.holdout.sharpe)} (dev ${fmt(s.dev_sharpe)}) · CAGR ${fmt(s.holdout.cagr,1)} · maxDD ${fmt(s.holdout.max_drawdown,1)} · <b class="${s.pass?"pass":"fail"}">${s.pass?"pass":"fail"}</b></div>`+
  chart([{v:cum(D.holdout.benchmark_returns),c:"var(--b)"},{v:cum(s.returns),c:"var(--a)"}],150,{base:1})+"</div>")})}
// table
const cols=[["hash","config"],["desc","description"],["family","family"],["sharpe","Sharpe"],["alpha_sharpe","alpha Sh."],["cagr","CAGR",1],["mdd","max DD",1],["trades","trades"],["win","win rate",1],["pf","PF"],["cost","cost/yr",1],["expo","exposure",1]];
$("#t thead tr").innerHTML=cols.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join("");
const opts={family:new Set(),signal:new Set(),filter:new Set(),exit:new Set(),sizing:new Set()};
D.trials.forEach(t=>Object.keys(opts).forEach(k=>opts[k].add(t[k])));
Object.entries(opts).forEach(([k,s])=>[...s].sort().forEach(v=>$("#f_"+k).insertAdjacentHTML("beforeend",`<option>${v}</option>`)));
let sortK="sharpe",sortD=-1;
function render(){const f={};Object.keys(opts).forEach(k=>f[k]=$("#f_"+k).value);const mdd=+$("#f_dd").value,mt=+$("#f_tr").value;
  let rows=D.trials.filter(t=>Object.keys(f).every(k=>!f[k]||t[k]===f[k])&&t.mdd<=mdd&&t.trades>=mt);
  rows.sort((a,b)=>((a[sortK]>b[sortK])-(a[sortK]<b[sortK]))*sortD);$("#count").textContent=rows.length+" match"+(rows.length>500?" (first 500 shown; filter or sort to see others)":"");
  $("#t tbody").innerHTML=rows.slice(0,500).map(t=>"<tr>"+cols.map(c=>`<td>${c[0]==="hash"?"<code>"+t.hash+"</code>":typeof t[c[0]]==="number"&&c[0]!=="trades"?fmt(t[c[0]],c[2]):t[c[0]]}</td>`).join("")+"</tr>").join("")}
document.querySelectorAll(".controls select,.controls input").forEach(e=>e.addEventListener("input",render));
document.querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{const k=th.dataset.k;sortD=k===sortK?-sortD:-1;sortK=k;render()}));
render();
</script></body></html>"""


def build_dashboard(reg: Registry, bench_returns: np.ndarray, score_start: int, study_id: str | None = None) -> str:
    st = reg.study(study_id)
    sid = st["study_id"]
    cfg = json.loads(st["config_json"])
    fams = reg.family_results(sid)
    meta = fams.pop("_benchmark", {})
    trials = reg.trials(st["dev_fp"], st["cost_fp"])
    rows = []
    for t in trials.itertuples():
        spec = json.loads(t.spec_json)
        m = json.loads(t.metrics_json)
        rows.append({"hash": t.spec_hash, "family": t.family, "signal": spec["signal"]["name"],
                     "filter": spec["filter"]["name"], "exit": spec["exit"]["name"], "sizing": spec["sizing"]["name"],
                     "desc": html.escape(_short(spec)), "sharpe": m.get("sharpe"), "alpha_sharpe": m.get("alpha_sharpe"),
                     "cagr": m.get("cagr"), "mdd": m.get("max_drawdown"), "trades": m.get("n_trades"),
                     "win": m.get("win_rate"), "pf": m.get("profit_factor") if m.get("profit_factor") != float("inf") else None,
                     "cost": m.get("cost_drag"), "expo": m.get("exposure")})
    cands = []
    cdf = reg.candidates(sid)
    if not cdf.empty:
        hashes, M = reg.trial_returns(trials[trials["spec_hash"].isin(cdf["spec_hash"])])
        col = {h: i for i, h in enumerate(hashes)}
        for c in cdf.itertuples():
            rob = json.loads(c.robustness_json)
            b = rob["base"]
            cands.append({"hash": c.spec_hash, "family": c.family, "desc": html.escape(rob["description"]),
                          "status": c.status, "failed": json.loads(c.failed_json), "sharpe": b["sharpe"],
                          "alpha_sharpe": b["alpha_sharpe"], "cagr": b["cagr"], "mdd": b["max_drawdown"],
                          "dsr": rob["dsr"]["dsr"],
                          "returns": np.round(M[score_start:, col[c.spec_hash]], 6).tolist()
                          if c.spec_hash in col else []})
    wf = [{"family": f, "returns": m["walk_forward"]["returns"], "start": m["walk_forward"]["start_row"],
           "sharpe": m["walk_forward"]["oos"]["sharpe"], "alpha_sharpe": m["walk_forward"]["oos_alpha_sharpe"],
           "bench_sharpe": m["walk_forward"]["benchmark_oos_sharpe"], "pbo": m["pbo"]["pbo"], "spa": m["spa"]["spa_p"]}
          for f, m in fams.items() if m.get("n_trials")]
    ho = reg.holdout_accesses(st["holdout_fp"])
    holdout = None
    if not ho.empty and ho.iloc[-1]["results_json"]:
        holdout = json.loads(ho.iloc[-1]["results_json"])
        holdout["contaminated"] = len(ho) > 1 or holdout.get("contaminated", False)
    data = {"study": {"id": sid, "name": st["name"], "split": json.loads(st["split_json"])},
            "synthetic": cfg["data"]["provider"] == "synthetic", "n_trials": meta.get("n_trials_total"),
            "bench_symbol": cfg["data"]["benchmark"], "bench": np.round(bench_returns, 6).tolist(),
            "score_start": score_start, "trials": rows, "cands": cands, "wf": wf, "holdout": holdout}
    payload = json.dumps(_clean(data)).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def _clean(o):
    """JSON for browsers: NaN and infinity become null."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    return o


def _short(spec: dict) -> str:
    def part(d):
        a = ",".join(str(v) for k, v in d.items() if k != "name")
        return f"{d['name']}({a})" if a else d["name"]
    return " · ".join(part(spec[k]) for k in ("signal", "filter", "exit", "sizing"))


def write_dashboard(reg: Registry, out_dir: str, bench_returns: np.ndarray, score_start: int,
                    study_id: str | None = None) -> Path:
    sid = reg.study(study_id)["study_id"]
    p = Path(out_dir) / f"dashboard_{sid}.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_dashboard(reg, bench_returns, score_start, study_id))
    return p
