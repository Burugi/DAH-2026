"""Self-contained HTML dashboard for one matchup (no CybORG, no CDN, no server).

    python dashboard.py <run_id|dir>     ->  <dir>/dashboard.html

The single HTML file embeds the episode data as JSON and draws, with plain canvas
JS: the fleet map animation (play/pause/scrub), a per-step tactic log (each side's
representative action + counts), and a running attack/defense score chart. Open it
in any browser. Reads only numpy + json so it works offline for the README quick
test.
"""
import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(SRC), "results")


def _resolve(arg):
    return arg if os.path.isdir(arg) else os.path.join(RESULTS, arg)


def _seed0(z, key, T, default_shape):
    if key in z.files:
        return np.asarray(z[key])[0][:T]
    return np.zeros(default_shape)


def build_dashboard(run_dir):
    run_dir = _resolve(run_dir)
    z = np.load(os.path.join(run_dir, "arrays.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    cfg = meta["config"]
    types = list(z["types"]); n = len(types); n_uav = types.count("uav")
    grid = cfg["fleet"]["grid"]

    pt, pr = z["pos_true"][0], z["pos_rep"][0]
    T = pt.shape[0]
    ro, lu = z["red_owned"][0], z["link_up"][0]
    ljam, lgps = z["label_jam"][0], z["label_gps"][0]
    dj = _seed0(z, "det_jam", T, (T, n)); dg = _seed0(z, "det_gps", T, (T, n))
    dc = _seed0(z, "det_comp", T, (T, n))
    det = (dj.astype(bool) | dg.astype(bool) | dc.astype(bool))
    red_act = _seed0(z, "red_act", T, (T,)).astype(int)
    blue_act = _seed0(z, "blue_act", T, (T,)).astype(int)
    red_cnt = z["red_cnt"][0][:T] if "red_cnt" in z.files else np.zeros((T, 1), int)
    blue_cnt = z["blue_cnt"][0][:T] if "blue_cnt" in z.files else np.zeros((T, 1), int)
    a_t = _seed0(z, "a_t", T, (T,)); d_t = _seed0(z, "d_t", T, (T,))

    names_red = meta.get("red_action_names", [str(i) for i in range(int(red_cnt.shape[1]))])
    names_blue = meta.get("blue_action_names", [str(i) for i in range(int(blue_cnt.shape[1]))])
    m = meta.get("metrics", {})

    def topk(cntrow, names, k=3):
        order = np.argsort(cntrow)[::-1]
        return [[names[i] if i < len(names) else str(i), int(cntrow[i])]
                for i in order[:k] if cntrow[i] > 0]

    frames = []
    for t in range(T):
        frames.append({
            "pt": [[round(float(x), 1), round(float(y), 1)] for x, y in pt[t]],
            "pr": [[round(float(x), 1), round(float(y), 1)] for x, y in pr[t]],
            "ro": [int(v) for v in ro[t]], "lu": [int(v) for v in lu[t]],
            "jam": [int(v) for v in ljam[t]], "gps": [int(v) for v in lgps[t]],
            "det": [int(v) for v in det[t]],
            "ra": names_red[red_act[t]] if red_act[t] < len(names_red) else str(red_act[t]),
            "ba": names_blue[blue_act[t]] if blue_act[t] < len(names_blue) else str(blue_act[t]),
            "rc": topk(red_cnt[t], names_red), "bc": topk(blue_cnt[t], names_blue),
            "a": round(float(a_t[t]), 3), "d": round(float(d_t[t]), 3),
        })

    data = {
        "name": cfg["name"], "red": meta.get("red_type", "?"), "blue": meta.get("blue_type", "?"),
        "grid": grid, "n": n, "n_uav": n_uav, "T": T,
        "types": types,
        "defense": meta.get("defense", {}),
        "scores": {"attack": m.get("attack_score"), "defense": m.get("defense_score"),
                   "availability": m.get("availability")},
        "frames": frames,
    }
    out = os.path.join(run_dir, "dashboard.html")
    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"dashboard -> {out}")
    return out


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>DroneSwarm dashboard</title>
<style>
 body{background:#12141c;color:#e6e6eb;font-family:Consolas,Menlo,monospace;margin:0;padding:14px}
 #wrap{display:flex;gap:14px;flex-wrap:wrap}
 #left{flex:0 0 auto}
 #right{flex:1 1 360px;min-width:340px;display:flex;flex-direction:column;gap:10px}
 canvas{background:#0e1018;border:1px solid #2a2e3a;border-radius:6px}
 .card{background:#181b24;border:1px solid #2a2e3a;border-radius:6px;padding:10px}
 h1{font-size:18px;margin:0 0 8px} h2{font-size:13px;margin:0 0 6px;color:#9aa0ad}
 .row{display:flex;align-items:center;gap:10px;margin:6px 0}
 button{background:#2a3346;color:#e6e6eb;border:1px solid #3a4358;border-radius:5px;padding:5px 12px;cursor:pointer}
 input[type=range]{flex:1}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td{padding:2px 4px} .k{color:#9aa0ad}
 .log{height:150px;overflow:auto;font-size:12px;line-height:1.5}
 .lg{display:inline-block;margin-right:12px}
 .dot{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:4px}
 .sq{display:inline-block;width:11px;height:11px;vertical-align:middle;margin-right:4px}
 .ring{background:transparent;border:2px solid}
</style></head><body>
<h1 id="ttl"></h1>
<div id="wrap">
 <div id="left"><canvas id="map" width="560" height="560"></canvas>
  <div class="card" style="margin-top:8px;font-size:12px">
   <span class="lg"><span class="dot" style="background:#4682eb"></span>friendly</span>
   <span class="lg"><span class="dot" style="background:#e14637"></span>compromised</span>
   <span class="lg"><span class="sq" style="background:#9b9ea8"></span>UGV (square)</span>
   <span class="lg">▲ UAV (triangle)</span>
   <span class="lg"><span class="dot ring" style="border-color:#b45ad2"></span>jammed</span>
   <span class="lg"><span style="color:#ffa500">→</span> GPS spoof</span>
   <span class="lg"><span class="dot ring" style="border-color:#ffe63c"></span>detected</span>
  </div>
 </div>
 <div id="right">
  <div class="card"><div class="row">
    <button id="play">▶ play</button>
    <input id="slider" type="range" min="0" value="0">
    <span id="stp"></span></div>
   <table id="info"></table></div>
  <div class="card"><h2>score over time (attack vs defense)</h2>
    <canvas id="chart" width="340" height="130"></canvas></div>
  <div class="card"><h2>tactic log (representative action + counts)</h2>
    <div id="now" style="font-size:13px;margin-bottom:6px"></div>
    <div class="log" id="log"></div></div>
 </div>
</div>
<script>
const D = /*__DATA__*/;
const MAP=560, PAD=18, g=D.grid, sc=(MAP-2*PAD)/g;
const F=(70,130,235), col={friend:"#4682eb",red:"#e14637",jam:"#b45ad2",spoof:"#ffa500",det:"#ffe63c",grid:"#272b36"};
const mc=document.getElementById("map").getContext("2d");
const cc=document.getElementById("chart").getContext("2d");
const sl=document.getElementById("slider"); sl.max=D.T-1;
function sx(x){return PAD + x*sc;} function sy(y){return MAP-PAD - y*sc;}
function drawMap(t){
  const f=D.frames[t];
  mc.fillStyle="#0e1018"; mc.fillRect(0,0,MAP,MAP);
  mc.strokeStyle=col.grid; mc.lineWidth=1;
  for(let gx=0; gx<=g; gx+=20){mc.beginPath();mc.moveTo(sx(gx),sy(0));mc.lineTo(sx(gx),sy(g));mc.stroke();
    mc.beginPath();mc.moveTo(sx(0),sy(gx));mc.lineTo(sx(g),sy(gx));mc.stroke();}
  for(let e=0;e<D.n;e++){
    const p=f.pt[e], x=sx(p[0]), y=sy(p[1]);
    const color = f.ro[e]? col.red: col.friend;
    if(f.gps[e]){ // spoof arrow, clamp reported into grid
      let rx=Math.max(0,Math.min(g,f.pr[e][0])), ry=Math.max(0,Math.min(g,f.pr[e][1]));
      mc.strokeStyle=col.spoof; mc.lineWidth=2; mc.beginPath();mc.moveTo(x,y);mc.lineTo(sx(rx),sy(ry));mc.stroke();
      mc.beginPath();mc.arc(sx(rx),sy(ry),4,0,7);mc.stroke();
    }
    mc.fillStyle=color;
    if(D.types[e]=="uav"){mc.beginPath();mc.moveTo(x,y-9);mc.lineTo(x-9,y+9);mc.lineTo(x+9,y+9);mc.closePath();mc.fill();}
    else{mc.fillRect(x-9,y-9,18,18);}
    if(f.jam[e]){mc.strokeStyle=col.jam;mc.lineWidth=2;mc.beginPath();mc.arc(x,y,13,0,7);mc.stroke();}
    if(f.det[e]){mc.strokeStyle=col.det;mc.lineWidth=1;mc.beginPath();mc.arc(x,y,17,0,7);mc.stroke();}
  }
}
function drawChart(t){
  const W=340,H=130,pad=22; cc.clearRect(0,0,W,H);
  cc.strokeStyle="#3a4150"; cc.lineWidth=1; cc.strokeRect(pad,6,W-pad-6,H-pad-6);
  cc.fillStyle="#9aa0ad"; cc.font="10px monospace";
  cc.fillText("1.0",2,12); cc.fillText("0.0",2,H-pad);
  const xx=i=>pad+(i/(D.T-1))*(W-pad-6), yy=v=>6+(1-v)*(H-pad-6);
  function line(key,color){cc.strokeStyle=color;cc.lineWidth=2;cc.beginPath();
    for(let i=0;i<=t;i++){const v=D.frames[i][key]; i==0?cc.moveTo(xx(i),yy(v)):cc.lineTo(xx(i),yy(v));}cc.stroke();}
  line("d", col.friend); line("a", col.red);
  cc.fillStyle=col.red; cc.fillText("attack A_t",W-92,16);
  cc.fillStyle=col.friend; cc.fillText("defense D_t",W-92,28);
}
function fmt(arr){return arr.map(x=>x[0]+"×"+x[1]).join(", ")||"-";}
function update(t){
  t=+t; drawMap(t); drawChart(t);
  const f=D.frames[t];
  document.getElementById("stp").textContent="step "+t+" / "+(D.T-1);
  document.getElementById("info").innerHTML=
    "<tr><td class=k>scenario</td><td>"+D.name+"</td><td class=k>defense</td><td>"+
      (D.defense.detector||"none")+"/"+(D.defense.response||"none")+"</td></tr>"+
    "<tr><td class=k>red (attack)</td><td>"+D.red+"</td><td class=k>blue (defense)</td><td>"+D.blue+"</td></tr>"+
    "<tr><td class=k>compromised</td><td>"+f.ro.reduce((a,b)=>a+b,0)+"/"+D.n+"</td>"+
      "<td class=k>episode A / D</td><td>"+D.scores.attack+" / "+D.scores.defense+"</td></tr>"+
    "<tr><td class=k>A_t / D_t</td><td>"+f.a+" / "+f.d+"</td>"+
      "<td class=k>availability</td><td>"+D.scores.availability+"</td></tr>";
  document.getElementById("now").innerHTML=
    "<b style='color:"+col.red+"'>RED</b> "+f.ra+" &nbsp;["+fmt(f.rc)+"] &nbsp;&nbsp;"+
    "<b style='color:"+col.friend+"'>BLUE</b> "+f.ba+" &nbsp;["+fmt(f.bc)+"]";
  let log="";
  for(let i=Math.max(0,t-9);i<=t;i++){const ff=D.frames[i];
    log="<div>t"+i+"  R:"+ff.ra+"  B:"+ff.ba+"  (A "+ff.a+" / D "+ff.d+")</div>"+log;}
  document.getElementById("log").innerHTML=log;
  sl.value=t;
}
let t=0, playing=false, timer=null;
document.getElementById("ttl").textContent="DroneSwarm — "+D.name+"  ·  red="+D.red+"  vs  blue="+D.blue;
sl.oninput=()=>{t=+sl.value;update(t);};
document.getElementById("play").onclick=function(){
  playing=!playing; this.textContent=playing?"⏸ pause":"▶ play";
  if(playing){timer=setInterval(()=>{t=(t+1)%D.T;update(t);},220);}else{clearInterval(timer);}
};
update(0);
</script></body></html>"""


if __name__ == "__main__":
    build_dashboard(sys.argv[1])
