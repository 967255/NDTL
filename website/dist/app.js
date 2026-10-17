(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const C = { bg: '#080b10', rule: '#262d37', text: '#edece8', muted: '#aaadb5', slate: '#a3b2c6', lavender: '#b4a1c7', gold: '#dbb57a' };
  const clusterColors = ['#c5a2ff','#6edcb5','#f497a8','#f1c16c','#72c3f4','#f0e2c5','#e884df'];
  const systemName = name => ({Lv:'Lv’s',Chen:'Chen’s'}[name] || name);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 3) => finite(value) ? value.toFixed(digits).replace(/^-/, '−') : '—';
  const svgEl = (name, attrs = {}, text) => { const el = document.createElementNS('http://www.w3.org/2000/svg', name); for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value); if (text !== undefined) el.textContent = text; return el; };
  const append = (parent, name, attrs, text) => { const el = svgEl(name, attrs, text); parent.append(el); return el; };
  const pretty = id => String(id).replace(/_/g, ' ').replace(/\b\w/g, s => s.toUpperCase()).replace('Lv', 'Lv');
  let data, currentCase, currentFrame = 2, selectedSystem = 0, mapMode = 'embedding', selectedLandscape = 0;
  const views = {};

  class Trajectory {
    constructor(canvas, color) {
      this.canvas = canvas; this.color = color; this.points = []; this.yaw = -.35; this.pitch = .25;
      canvas.dataset.rotatable = 'true';
      this.observer = new ResizeObserver(() => this.draw()); this.observer.observe(canvas);
      let drag = null;
      canvas.addEventListener('pointerdown', event => { drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); });
      canvas.addEventListener('pointermove', event => { if (!drag) return; this.yaw += (event.clientX - drag.x) * .007; this.pitch += (event.clientY - drag.y) * .005; drag = { x: event.clientX, y: event.clientY }; this.draw(); });
      for (const eventName of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(eventName, () => { drag = null; });
    }
    set(points) {
      this.points = (points || []).filter(p => p.length === 3 && p.every(finite));
      if (this.points.length) {
        const lo = [0,1,2].map(k => Math.min(...this.points.map(p => p[k]))), hi = [0,1,2].map(k => Math.max(...this.points.map(p => p[k])));
        const span = Math.max(...hi.map((v,k) => v-lo[k])) || 1;
        this.normalized = this.points.map(p => p.map((v,k) => (v-(lo[k]+hi[k])/2)/span));
      } else this.normalized = [];
      this.draw();
    }
    draw() {
      const rect = this.canvas.getBoundingClientRect(); if (!rect.width || !rect.height) return;
      const w = rect.width, h = rect.height, ratio = Math.min(devicePixelRatio || 1, 2), pw = Math.round(w*ratio), ph = Math.round(h*ratio);
      if (this.canvas.width !== pw || this.canvas.height !== ph) { this.canvas.width=pw; this.canvas.height=ph; }
      const ctx = this.canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,w,h);
      if (!this.normalized?.length) return;
      const cy=Math.cos(this.yaw), sy=Math.sin(this.yaw), cp=Math.cos(this.pitch), sp=Math.sin(this.pitch);
      const projected = this.normalized.map(([x,y,z]) => { const a=x*cy-y*sy,b=x*sy+y*cy; return [a,-z*cp+b*sp]; });
      const xs=projected.map(p=>p[0]),ys=projected.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
      const scale=Math.min((w-24)/Math.max(x1-x0,.01),(h-35)/Math.max(y1-y0,.01))*.88;
      const thumbnail=this.canvas.closest('.orbit-tile');ctx.lineWidth=thumbnail ? .72 : (this.color===C.gold?1.65:1.2);ctx.strokeStyle=this.color;ctx.globalAlpha=thumbnail ? .74 : .88;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();
      projected.forEach(([x,y],i)=>{const px=w/2+(x-(x0+x1)/2)*scale,py=h/2+(y-(y0+y1)/2)*scale;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.globalAlpha=1;
    }
  }

  function niceTicks(min,max,n=4) {
    if (max<=min) return [min];
    const raw=(max-min)/n, power=10**Math.floor(Math.log10(raw)), step=[1,2,2.5,5,10].map(x=>x*power).find(x=>x>=raw)||power*10;
    const result=[];for(let value=Math.ceil(min/step)*step;value<=max+step*.001;value+=step)result.push(Math.abs(value)<step*1e-9?0:value);
    return result;
  }
  function plot(svg, series, options={}) {
    const w=Math.max(240,svg.getBoundingClientRect().width),h=svg.getBoundingClientRect().height||240,m={l:53,r:18,t:20,b:43};
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    append(svg,'title',{},options.title||svg.getAttribute('aria-label')||'Data plot');
    const valid=series.flatMap(s=>s.points.filter(p=>finite(p.x)&&finite(p.y)));
    if(!valid.length){append(svg,'text',{x:w/2,y:h/2,'text-anchor':'middle',class:'plot-label'},'No numerical data available');return;}
    const xVal=x=>options.logX?Math.log10(x):x;
    let xmin=options.xDomain?.[0]??Math.min(...valid.map(p=>p.x)),xmax=options.xDomain?.[1]??Math.max(...valid.map(p=>p.x));
    let ymin=Math.min(...valid.map(p=>finite(p.lo)?p.lo:p.y)),ymax=Math.max(...valid.map(p=>finite(p.hi)?p.hi:p.y));
    if(options.includeZero){ymin=Math.min(0,ymin);ymax=Math.max(0,ymax);}
    const padding=(ymax-ymin||Math.abs(ymax)||1)*.13;ymin-=padding;ymax+=padding;
    const sx=x=>m.l+(xVal(x)-xVal(xmin))/(xVal(xmax)-xVal(xmin)||1)*(w-m.l-m.r),sy=y=>h-m.b-(y-ymin)/(ymax-ymin)*(h-m.t-m.b);
    for(const y of niceTicks(ymin,ymax,4)) {append(svg,'line',{x1:m.l,y1:sy(y),x2:w-m.r,y2:sy(y),class:'plot-grid'});append(svg,'text',{x:m.l-9,y:sy(y)+4,'text-anchor':'end',class:'plot-axis'},Math.abs(y)>=100?y.toFixed(0):Number(y.toPrecision(3)).toString().replace('-','−'));}
    const ticks=options.logX?[.001,.003,.01,.03,.1,.3,1].filter(x=>x>=xmin&&x<=xmax):(options.xTicks||(w<390?[xmin,(xmin+xmax)/2,xmax]:[xmin,xmin+(xmax-xmin)/4,(xmin+xmax)/2,xmin+3*(xmax-xmin)/4,xmax]));
    for(const x of ticks)append(svg,'text',{x:sx(x),y:h-m.b+20,'text-anchor':'middle',class:'plot-axis'},Number(x.toPrecision(3)).toString());
    append(svg,'rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b,fill:'none',stroke:C.rule});
    if(ymin<0&&ymax>0)append(svg,'line',{x1:m.l,y1:sy(0),x2:w-m.r,y2:sy(0),stroke:C.muted,'stroke-width':.8,'stroke-dasharray':'3 4'});
    if(finite(options.selected))append(svg,'line',{x1:sx(options.selected),x2:sx(options.selected),y1:m.t,y2:h-m.b,stroke:C.muted,'stroke-width':.8,'stroke-dasharray':'3 4'});
    for(const s of series){
      const points=s.points.filter(p=>finite(p.x)&&finite(p.y));if(!points.length)continue;
      if(points.every(p=>finite(p.lo)&&finite(p.hi)))append(svg,'polygon',{points:[...points.map(p=>`${sx(p.x)},${sy(p.lo)}`),...points.slice().reverse().map(p=>`${sx(p.x)},${sy(p.hi)}`)].join(' '),fill:s.color,opacity:.12});
      append(svg,'polyline',{points:points.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':1.6});
      for(const p of points){const circle=append(svg,'circle',{cx:sx(p.x),cy:sy(p.y),r:p.x===options.selected?4:2.5,fill:s.color,stroke:C.bg,'stroke-width':1});append(circle,'title',{},`${s.label}: ${fmt(p.y,4)} at ${p.x}`);}
    }
    append(svg,'text',{x:m.l+(w-m.l-m.r)/2,y:h-5,'text-anchor':'middle',class:'plot-label'},options.xLabel||'Fusion weight α');
    if(options.onSelect){const overlay=append(svg,'rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b,fill:'transparent',style:'cursor:crosshair'});overlay.addEventListener('click',event=>{const bound=svg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-bound.left-m.l)/(w-m.l-m.r)));options.onSelect(xmin+fraction*(xmax-xmin));});}
  }

  let reverseField=false;
  function renderFeatures(){
    const field=[[],[],[]];for(let y=0;y<8;y++)for(let x=0;x<12;x++){const X=(x+.5)/12,Y=(y+.5)/8,Z=.5;const v=[10*(Y-X),(X-.5)*(28-50*Z)-(Y-.5),50*(X-.5)*(Y-.5)-(8/3)*Z];v.forEach((value,k)=>field[k].push(value));}
    const gram=field.map(a=>field.map(b=>a.reduce((sum,value,index)=>sum+value*b[index],0)/a.length));
    const paint=(svg,rows,gramMode)=>{const w=Math.max(150,svg.getBoundingClientRect().width),h=svg.getBoundingClientRect().height||200,left=gramMode?8:27,right=gramMode?8:5,top=15,bottom=28,cols=rows[0].length,cw=(w-left-right)/cols,ch=(h-top-bottom)/rows.length,max=Math.max(...rows.flat().map(Math.abs))||1;
      svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);append(svg,'title',{},gramMode?'Input-layer Gram matrix, unchanged by time reversal':'Three vector-field channels over 96 spatial samples');
      rows.forEach((row,r)=>{if(!gramMode)append(svg,'text',{x:18,y:top+(r+.5)*ch,'text-anchor':'end',class:'plot-axis'},['ẋ','ẏ','ż'][r]);row.forEach((raw,c)=>{const value=raw*(gramMode?1:(reverseField?-1:1)),rect=append(svg,'rect',{x:left+c*cw,y:top+r*ch,width:Math.max(.2,cw-.5),height:ch-.6,fill:value>=0?C.gold:C.slate,opacity:.08+.85*Math.sqrt(Math.abs(value)/max)});append(rect,'title',{},`${gramMode?'G':'F'}[${r+1}, ${c+1}] = ${fmt(value,4)}`);if(gramMode)append(svg,'text',{x:left+(c+.5)*cw,y:top+(r+.5)*ch+4,'text-anchor':'middle',fill:C.text,'font-size':11},fmt(value,1));});});append(svg,'text',{x:(w+left)/2,y:h-6,'text-anchor':'middle',class:'plot-label'},gramMode?'FFᵀ / N':'Spatial sample index');};
    paint($('content-map'),field,false);paint($('gram-map'),gram,true);$('reverse-field').setAttribute('aria-pressed',String(reverseField));$('reverse-field').textContent=reverseField?'Restore original field':'Reverse time: F → −F';$('feature-status').textContent=reverseField?'Input layer · F changes sign · Gram difference = 0':'Input layer · Gram unchanged under F → −F';
  }

  let orbitViews=[], landscapeYaw=-.6, landscapePitch=.9, landscapeHits=[], selectedA='chen', selectedB='lv';
  let landscapeContours=null;
  let pairLookup, systemLookup;
  const caseById=id=>data.fusion.cases.find(item=>item.id===id);
  const pairKey=(a,b)=>[a,b].sort().join('|');
  const frameValid=frame=>frame.status!=='diverged'&&frame.trajectory.length>0;

  function ribbon(svg,series){
    const w=svg.getBoundingClientRect().width,h=svg.getBoundingClientRect().height,m={l:36,r:12,t:17,b:25};
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    const values=series.flatMap(s=>s.points.filter(p=>finite(p.y)).map(p=>p.y));if(!values.length)return;
    let lo=Math.min(...values),hi=Math.max(...values),pad=(hi-lo||1)*.18;lo-=pad;hi+=pad;
    const sx=x=>m.l+x*(w-m.l-m.r),sy=y=>h-m.b-(y-lo)/(hi-lo)*(h-m.t-m.b);
    for(const y of [Math.min(...values),Math.max(...values)])append(svg,'text',{x:m.l-9,y:sy(y)+3,'text-anchor':'end',class:'plot-axis'},fmt(y,2));
    append(svg,'line',{x1:m.l,x2:w-m.r,y1:h-m.b,y2:h-m.b,stroke:C.rule});
    if(lo<0&&hi>0)append(svg,'line',{x1:m.l,x2:w-m.r,y1:sy(0),y2:sy(0),stroke:C.muted,'stroke-width':.5,'stroke-dasharray':'2 5'});
    const f=currentCase.frames[currentFrame];append(svg,'line',{x1:sx(f.weight),x2:sx(f.weight),y1:m.t-5,y2:h-m.b,stroke:C.gold,'stroke-width':.7,opacity:.5,'stroke-dasharray':'2 4'});
    for(const s of series){
      let group=[];const groups=[];for(const p of s.points){if(finite(p.y))group.push(p);else if(group.length){groups.push(group);group=[];}}if(group.length)groups.push(group);
      for(const points of groups){
        if(s.area&&points.length>1)append(svg,'polygon',{points:[`${sx(points[0].x)},${h-m.b}`,...points.map(p=>`${sx(p.x)},${sy(p.y)}`),`${sx(points.at(-1).x)},${h-m.b}`].join(' '),fill:s.color,opacity:.065});
        append(svg,'polyline',{points:points.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':1.6,'stroke-linejoin':'round','stroke-dasharray':s.dashed?'3 5':''});
        for(const p of points){const active=p.x===f.weight,circle=append(svg,'circle',{cx:sx(p.x),cy:sy(p.y),r:active?4.2:2.5,fill:active?s.color:C.bg,stroke:s.color,'stroke-width':1.2});append(circle,'title',{},`${s.label}: ${fmt(p.y,4)} · α=${p.x}`);}
      }
    }
    for(const x of [0,.5,1])append(svg,'text',{x:sx(x),y:h-4,'text-anchor':'middle',class:'plot-axis'},x);
    const overlay=append(svg,'rect',{x:m.l,y:0,width:w-m.l-m.r,height:h,fill:'transparent',style:'cursor:crosshair'});
    const select=event=>{const alpha=(event.clientX-svg.getBoundingClientRect().left-m.l)/(w-m.l-m.r);const index=currentCase.frames.reduce((best,frame,i)=>Math.abs(frame.weight-alpha)<Math.abs(currentCase.frames[best].weight-alpha)?i:best,0);if(index!==currentFrame)selectFrame(index);};
    overlay.addEventListener('pointermove',event=>{if(event.pointerType==='mouse')select(event);});overlay.addEventListener('click',select);
  }
  function renderSignatures(){
    const frames=currentCase.frames,f=frames[currentFrame],valid=frameValid(f);
    $('selected-weight').textContent=Number(f.weight.toFixed(3));$('selected-mle').textContent=valid?fmt(f.metrics.lyapunov[0]):'—';$('selected-ky').textContent=valid?fmt(f.metrics.kaplanYorke):'—';
    const points=key=>frames.map(frame=>({x:frame.weight,y:frameValid(frame)?key(frame.metrics):null}));
    ribbon($('lyapunov-plot'),[{label:'λ₁',color:C.gold,area:true,points:points(m=>m.lyapunov[0])}]);
    const dims=[{label:'Kaplan–Yorke',color:C.lavender,points:points(m=>m.kaplanYorke)}];
    if(frames.some(frame=>frameValid(frame)&&finite(frame.metrics.intrinsicDimension)))dims.push({label:'Intrinsic dimension',color:C.slate,dashed:true,points:points(m=>m.intrinsicDimension)});
    $('dimension-key').textContent=dims.length>1?'Kaplan–Yorke · intrinsic (dashed)':'Kaplan–Yorke dimension';ribbon($('dimension-plot'),dims);
    $('fusion-equations').textContent=f.equations.join('\n\n');$('fusion-spectrum').textContent=valid?'Lyapunov spectrum: '+f.metrics.lyapunov.map(v=>fmt(v)).join(' · '):'Archived simulation flagged as divergent; this endpoint is not interpreted as a bounded system.';
  }
  function selectFrame(index){currentFrame=index;document.querySelectorAll('.orbit-tile').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));const scroll=document.querySelector('.fusion-scroll'),tile=document.querySelectorAll('.orbit-tile')[index];if(scroll.scrollWidth>scroll.clientWidth){const a=tile.getBoundingClientRect(),b=scroll.getBoundingClientRect();if(a.left<b.left||a.right>b.right)scroll.scrollTo({left:scroll.scrollLeft+a.left-b.left-(b.width-a.width)/2,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});}renderSignatures();}
  function setCase(id){
    currentCase=caseById(id);currentFrame=currentCase.frames.reduce((best,frame,index)=>Math.abs(frame.weight-.5)<Math.abs(currentCase.frames[best].weight-.5)?index:best,0);
    document.querySelectorAll('button[data-case]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.case===id)));$('fusion-panel').setAttribute('aria-labelledby',`tab-${id}`);
    orbitViews.forEach(view=>view.observer.disconnect());orbitViews=[];$('orbit-strip').replaceChildren();
    const c=currentCase;document.documentElement.dataset.case=id;$('fusion-from').textContent=systemName(c.parentB);$('fusion-to').textContent=systemName(c.parentA);
    $('fusion-axis-label').textContent='Fusion weight α';
    c.frames.forEach((frame,index)=>{
      const button=document.createElement('button');button.type='button';button.className='orbit-tile'+(frameValid(frame)?'':' is-failed');button.dataset.frame=index;button.setAttribute('aria-pressed',String(index===currentFrame));button.setAttribute('aria-label',`α=${frame.weight}${frameValid(frame)?'':', divergent evaluation'}`);
      const canvas=document.createElement('canvas');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`Trajectory at α=${frame.weight}`);button.append(canvas);
      const label=document.createElement('span');label.className='orbit-weight';label.textContent=Number(frame.weight.toFixed(3));button.append(label);$('orbit-strip').append(button);
      const color=index===0?C.lavender:index===c.frames.length-1?C.slate:C.gold,view=new Trajectory(canvas,color);view.set(frame.trajectory);orbitViews.push(view);
      canvas.removeAttribute('data-rotatable');canvas.style.cursor='pointer';button.addEventListener('pointerenter',event=>{if(event.pointerType==='mouse')selectFrame(index);});button.addEventListener('click',()=>selectFrame(index));button.addEventListener('focus',()=>selectFrame(index));
      button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const next=(index+(event.key==='ArrowRight'?1:-1)+c.frames.length)%c.frames.length;document.querySelectorAll('.orbit-tile')[next].focus();});
    });
    $('case-source').textContent='9 weights';
    document.querySelector('.fusion-caption span:last-child').textContent=innerWidth<=760?'Swipe to explore · select a trajectory':'Hover or select a trajectory';
    $('case-note').textContent=c.paperReference+'. '+c.frames[currentFrame].trajectoryNote+' '+(c.alphaMeaning||'α is the normalized content share; content–style weights are nonuniform.')+' Each view is fitted independently.';
    renderSignatures();
  }
  function renderLayer(){
    const metric=$('layer-metric').value==='ky'?'ky_dim_hat':'lambda1_hat';
    plot($('layer-plot'),[0,1,2].map((layer,index)=>({label:`Layer ${layer+1}`,color:[C.slate,C.lavender,C.gold][index],points:data.features.summaries.filter(row=>row.layer_id===layer).sort((a,b)=>a.epsilon_nominal-b.epsilon_nominal).map(row=>({x:row.epsilon_nominal,y:row[metric].mean,lo:row[metric].q25,hi:row[metric].q75}))})),{logX:true,xLabel:'Perturbation amplitude ε'});
  }
  function canvasContext(canvas){const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return {ctx,w:rect.width,h:rect.height};}
  function terrainColor(t){
    const stops=[[0,[128,169,200]],[.25,[182,176,217]],[.6,[191,149,199]],[1,[219,181,122]]];
    t=Math.max(0,Math.min(1,t));const index=stops.findIndex(stop=>stop[0]>=t),upper=stops[Math.max(1,index)],lower=stops[Math.max(0,index-1)],fraction=(t-lower[0])/(upper[0]-lower[0]);
    return lower[1].map((value,k)=>Math.round(value+(upper[1][k]-value)*fraction));
  }
  function buildLandscapeContours(surface,levels){
    return levels.map(level=>{
      const segments=[];
      for(let row=0;row<surface.y.length-1;row++)for(let col=0;col<surface.x.length-1;col++){
        const corners=[[surface.x[col],surface.y[row],surface.z[row][col]],[surface.x[col+1],surface.y[row],surface.z[row][col+1]],[surface.x[col+1],surface.y[row+1],surface.z[row+1][col+1]],[surface.x[col],surface.y[row+1],surface.z[row+1][col]]],crossings=new Map();
        for(let edge=0;edge<4;edge++){
          const a=corners[edge],b=corners[(edge+1)%4];if((a[2]<level)===(b[2]<level))continue;
          const fraction=(level-a[2])/(b[2]-a[2]);crossings.set(edge,[a[0]+(b[0]-a[0])*fraction,a[1]+(b[1]-a[1])*fraction,level]);
        }
        if(crossings.size===2)segments.push([...crossings.values()]);
        else if(crossings.size===4){
          const centerHigh=corners.reduce((sum,p)=>sum+p[2],0)/4>=level,cornerZeroHigh=corners[0][2]>=level;
          const edges=centerHigh===cornerZeroHigh?[[0,1],[2,3]]:[[3,0],[1,2]];
          for(const [a,b] of edges)segments.push([crossings.get(a),crossings.get(b)]);
        }
      }
      return {level,segments};
    });
  }
  function renderLandscape(){
    const {ctx,w,h}=canvasContext($('systems-landscape')),source=data.landscape,surface=source.surface;
    const minU=Math.min(...source.systems.map(s=>s.U)),maxU=Math.max(...source.systems.map(s=>s.U)),height=u=>(u-minU)/(maxU-minU)*1.2;
    const cy=Math.cos(landscapeYaw),sy=Math.sin(landscapeYaw),cp=Math.cos(landscapePitch),sp=Math.sin(landscapePitch);
    const raw=(x,y,z)=>{const a=x*cy-y*sy,b=x*sy+y*cy;return [a,b*cp-z*sp,b*sp+z*cp];};
    const grid=surface.y.map((y,row)=>surface.x.map((x,col)=>raw(x,y,height(surface.z[row][col]))));
    const all=grid.flat(),minX=Math.min(...all.map(p=>p[0])),maxX=Math.max(...all.map(p=>p[0])),minY=Math.min(...all.map(p=>p[1])),maxY=Math.max(...all.map(p=>p[1]));
    const scale=Math.min((w-75)/(maxX-minX),(h-85)/(maxY-minY)),screen=p=>[w/2+(p[0]-(minX+maxX)/2)*scale,h/2+15+(p[1]-(minY+maxY)/2)*scale,p[2]];
    const projected=grid.map(row=>row.map(screen)),cells=[];
    for(let row=0;row<grid.length-1;row++)for(let col=0;col<grid[0].length-1;col++){
      const corners=[projected[row][col],projected[row][col+1],projected[row+1][col+1],projected[row+1][col]],u=(surface.z[row][col]+surface.z[row][col+1]+surface.z[row+1][col]+surface.z[row+1][col+1])/4,t=Math.max(0,Math.min(1,(u-minU)/(maxU-minU)));
      cells.push({corners,depth:corners.reduce((sum,p)=>sum+p[2],0)/4,t});
    }
    cells.sort((a,b)=>a.depth-b.depth);
    for(const cell of cells){const {corners,t}=cell;ctx.beginPath();corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=`rgba(${terrainColor(t).join(',')},${.29+.1*t})`;ctx.fill();ctx.strokeStyle='rgba(163,178,198,.09)';ctx.lineWidth=.45;ctx.stroke();}
    if(!landscapeContours)landscapeContours=buildLandscapeContours(surface,niceTicks(minU,maxU,12));
    let contourCount=0;
    for(const contour of landscapeContours){
      const t=(contour.level-minU)/(maxU-minU),color=terrainColor(t);ctx.beginPath();
      for(const segment of contour.segments){const a=screen(raw(segment[0][0],segment[0][1],height(contour.level))),b=screen(raw(segment[1][0],segment[1][1],height(contour.level)));ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);contourCount++;}
      ctx.strokeStyle=`rgba(${color.join(',')},.6)`;ctx.lineWidth=.8;ctx.stroke();
    }
    const systems=source.systems.map(system=>({...system,name:systemName(system.name),position:screen(raw(system.x,system.y,height(system.U)))})).sort((a,b)=>a.position[2]-b.position[2]);
    landscapeHits=[];
    for(const s of systems){const [x,y]=s.position,active=s.id===selectedA||s.id===selectedB,color=clusterColors[(s.cluster-1)%7];
      ctx.beginPath();ctx.arc(x,y,active?6.5:4.1,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle=active?C.text:C.bg;ctx.lineWidth=active?1.8:1.2;ctx.stroke();landscapeHits.push({x,y,system:s});
    }
    const a=systems.find(s=>s.id===selectedA),b=systems.find(s=>s.id===selectedB);
    if(a&&b){ctx.setLineDash([3,5]);ctx.lineWidth=1.1;ctx.strokeStyle='rgba(241,193,108,.9)';ctx.beginPath();ctx.moveTo(...a.position.slice(0,2));ctx.lineTo(...b.position.slice(0,2));ctx.stroke();ctx.setLineDash([]);}
    const labelRects=[];
    for(const [index,s] of [a,b].filter(Boolean).entries()){
      const [x,y]=s.position;ctx.font='12px Arial';const tw=ctx.measureText(s.name).width,tx=Math.max(8,Math.min(w-tw-8,x+(index?12:-tw-12)));let ty=Math.max(40,y+(index?19:-12));for(const box of labelRects)if(tx<box.x+box.w+5&&tx+tw>box.x-5&&ty>box.y-16&&ty-12<box.y+4)ty=Math.max(40,box.y-20);labelRects.push({x:tx,y:ty,w:tw});ctx.strokeStyle=C.bg;ctx.lineWidth=4;ctx.strokeText(s.name,tx,ty);ctx.fillStyle=C.text;ctx.fillText(s.name,tx,ty);
    }
    ctx.font='10px Arial';ctx.fillStyle=C.muted;ctx.textAlign='center';ctx.fillText('Spectral embedding',w/2,h-5);ctx.textAlign='left';
    $('systems-landscape').dataset.selectedPair=a&&b?pairKey(selectedA,selectedB):'';
    $('systems-landscape').dataset.selectedCount=Number(Boolean(a))+Number(Boolean(b));
    $('systems-landscape').dataset.contourSegments=contourCount;
    $('clear-landscape').disabled=!a&&!b;
    $('landscape-low').textContent=fmt(minU,2);$('landscape-high').textContent=fmt(maxU,2);
  }
  function renderPairPath(pair,reversed){
    const svg=$('pair-path'),w=svg.getBoundingClientRect().width,h=svg.getBoundingClientRect().height,m={l:38,r:10,t:12,b:21},gap=20,rh=(h-m.t-m.b-2*gap)/3;
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    const spectra=reversed?pair.spectra.slice().reverse():pair.spectra;
    for(let axis=0;axis<3;axis++){
      const values=spectra.filter(Boolean).map(s=>s[axis]);if(!values.length){append(svg,'text',{x:w/2,y:h/2,'text-anchor':'middle',class:'plot-label'},'No valid estimates');break;}let lo=Math.min(...values),hi=Math.max(...values),pad=(hi-lo||1)*.15;lo-=pad;hi+=pad;
      const top=m.t+axis*(rh+gap),sx=i=>m.l+i/8*(w-m.l-m.r),sy=value=>top+rh-(value-lo)/(hi-lo)*rh,color=[C.gold,C.slate,C.lavender][axis];
      append(svg,'text',{x:0,y:top+10,fill:color,'font-size':10},`λ${['₁','₂','₃'][axis]}`);
      for(const value of [Math.min(...values),Math.max(...values)])append(svg,'text',{x:m.l-7,y:sy(value)+3,'text-anchor':'end',class:'plot-axis','font-size':8},fmt(value,1));
      append(svg,'line',{x1:m.l,x2:w-m.r,y1:top+rh,y2:top+rh,stroke:C.rule,'stroke-width':.7});
      if(lo<0&&hi>0)append(svg,'line',{x1:m.l,x2:w-m.r,y1:sy(0),y2:sy(0),stroke:C.muted,'stroke-width':.5,'stroke-dasharray':'2 4'});
      let segment=[];const flush=()=>{if(segment.length)append(svg,'polyline',{points:segment.join(' '),stroke:color,fill:'none','stroke-width':1.3});segment=[];};
      spectra.forEach((s,i)=>{if(s){segment.push(`${sx(i)},${sy(s[axis])}`);const dot=append(svg,'circle',{cx:sx(i),cy:sy(s[axis]),r:2.1,fill:color});append(dot,'title',{},`α=${data.landscape.alphas[i]} · λ${axis+1}=${fmt(s[axis],4)}`);}else{flush();append(svg,'text',{x:sx(i),y:top+rh/2+4,'text-anchor':'middle',fill:'#ba9999','font-size':12,'data-failure':i},'×');}});flush();
    }
    for(const x of [0,.5,1])append(svg,'text',{x:m.l+x*(w-m.l-m.r),y:h-2,'text-anchor':'middle',class:'plot-axis'},x);
  }
  function renderMatrix(){
    const svg=$('distance-matrix'),w=svg.getBoundingClientRect().width,h=svg.getBoundingClientRect().height,source=data.landscape,order=source.clusterOrder,n=order.length,size=Math.min(w-16,h-27),left=(w-size)/2+4,top=10,cell=size/n;
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
    for(let row=0;row<n;row++)for(let col=0;col<n;col++){
      const a=source.systems[order[row]],b=source.systems[order[col]],d=source.distanceMatrix[order[row]][order[col]],color=`rgb(${Math.round(18+201*d)},${Math.round(25+156*d)},${Math.round(34+88*d)})`;
      const rect=append(svg,'rect',{x:left+col*cell,y:top+row*cell,width:cell+.15,height:cell+.15,fill:color,'data-cell':`${a.id}|${b.id}`});append(rect,'title',{},`${systemName(a.name)} + ${systemName(b.name)} · d=${fmt(d,4)}`);
      if(row===0){append(svg,'rect',{x:left+col*cell,y:top-7,width:cell+.1,height:3,fill:clusterColors[(b.cluster-1)%7]});append(svg,'rect',{x:left-7,y:top+col*cell,width:3,height:cell+.1,fill:clusterColors[(b.cluster-1)%7]});}
    }
    const ia=order.findIndex(i=>source.systems[i].id===selectedA),ib=order.findIndex(i=>source.systems[i].id===selectedB);
    if(ia>=0&&ib>=0){for(const [row,col] of [[ia,ib],[ib,ia]])append(svg,'rect',{x:left+col*cell-.7,y:top+row*cell-.7,width:cell+1.4,height:cell+1.4,fill:'none',stroke:C.text,'stroke-width':1.2,'pointer-events':'none'});}
    else if(ia>=0||ib>=0){const index=Math.max(ia,ib);append(svg,'rect',{x:left,y:top+index*cell,width:size,height:cell,fill:'none',stroke:C.text,opacity:.7,'stroke-width':.8,'pointer-events':'none'});append(svg,'rect',{x:left+index*cell,y:top,width:cell,height:size,fill:'none',stroke:C.text,opacity:.7,'stroke-width':.8,'pointer-events':'none'});}
    append(svg,'text',{x:left,y:top+size+16,class:'plot-axis'},'low d');append(svg,'text',{x:left+size,y:top+size+16,'text-anchor':'end',class:'plot-axis'},'high d');
    svg.onclick=event=>{const rect=svg.getBoundingClientRect(),col=Math.floor((event.clientX-rect.left-left)/cell),row=Math.floor((event.clientY-rect.top-top)/cell);if(row<0||col<0||row>=n||col>=n||row===col)return;setPair(source.systems[order[row]].id,source.systems[order[col]].id);};
  }
  function setPair(a,b){
    selectedA=systemLookup.has(a)?a:null;selectedB=systemLookup.has(b)?b:null;if(selectedA&&selectedA===selectedB)selectedB=null;
    $('landscape-tooltip').hidden=true;renderPair();
  }
  function toggleLandscapePoint(id){
    if(id===selectedA)setPair(null,selectedB);
    else if(id===selectedB)setPair(selectedA,null);
    else if(!selectedA)setPair(id,selectedB);
    else if(!selectedB)setPair(selectedA,id);
    else setPair(selectedB,id);
  }
  function renderPair(){
    const pair=selectedA&&selectedB?pairLookup.get(pairKey(selectedA,selectedB)):null;
    document.documentElement.dataset.pair=pair?pair.id:'';$('system-a').value=selectedA||'';$('system-b').value=selectedB||'';
    $('path-a').textContent=selectedA?systemLookup.get(selectedA).name:'—';$('path-b').textContent=selectedB?systemLookup.get(selectedB).name:'—';
    $('pair-success').textContent=pair?`${pair.validCount} / 9 valid`:(selectedA||selectedB?'Select one more system':'Select two systems');
    document.querySelectorAll('button[data-pair]').forEach(button=>button.setAttribute('aria-pressed',String(Boolean(pair&&button.dataset.pair===pair.id))));
    $('path-fingerprint').innerHTML=['Endpoint gap','Path deviation','Roughness','Chaos loss'].map((label,index)=>`<div><div class="fingerprint-label"><span>${label}</span><span>${pair?fmt(pair.ranks[index],2):'—'}</span></div><div class="fingerprint-track">${pair?`<i style="width:${pair.ranks[index]*100}%"></i>`:''}</div></div>`).join('');$('pair-distance').textContent=pair?fmt(pair.distance):'—';
    if(pair)renderPairPath(pair,pair.a!==selectedA);
    else{const svg=$('pair-path'),w=svg.getBoundingClientRect().width,h=svg.getBoundingClientRect().height;svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);append(svg,'text',{x:w/2,y:h/2,'text-anchor':'middle',class:'plot-label'},'Select two systems');}
    renderMatrix();renderLandscape();
  }
  function attachLandscape(){
    const canvas=$('systems-landscape');let drag=null,drawFrame=0;
    const findPoint=event=>{const rect=canvas.getBoundingClientRect(),radius=event.pointerType==='touch'?18:12;let nearest=null,best=radius;for(const hit of landscapeHits){const distance=Math.hypot(hit.x-(event.clientX-rect.left),hit.y-(event.clientY-rect.top));if(distance<=best){best=distance;nearest=hit;}}return nearest;};
    const requestDraw=()=>{if(!drawFrame)drawFrame=requestAnimationFrame(()=>{drawFrame=0;renderLandscape();});};
    canvas.addEventListener('pointerdown',event=>{if(event.button!==0||!event.isPrimary)return;drag={pointerId:event.pointerId,startX:event.clientX,startY:event.clientY,x:event.clientX,y:event.clientY,moved:false};canvas.focus({preventScroll:true});canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener('pointermove',event=>{
      if(drag&&event.pointerId===drag.pointerId){
        if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<6)return;
        drag.moved=true;landscapeYaw+=(event.clientX-drag.x)*.006;landscapePitch=Math.max(.35,Math.min(1.3,landscapePitch+(event.clientY-drag.y)*.004));drag.x=event.clientX;drag.y=event.clientY;$('landscape-tooltip').hidden=true;requestDraw();return;
      }
      const hit=findPoint(event);$('landscape-tooltip').hidden=!hit;if(hit)$('landscape-tooltip').textContent=`${hit.system.name} · U=${fmt(hit.system.U)} · ${hit.system.id===selectedA||hit.system.id===selectedB?'Deselect':'Select'}`;
    });
    canvas.addEventListener('pointerup',event=>{if(!drag||event.pointerId!==drag.pointerId)return;const wasDrag=drag.moved||Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>=6;drag=null;if(wasDrag)return;const hit=findPoint(event);if(hit)toggleLandscapePoint(hit.system.id);else setPair(null,null);});
    for(const name of ['pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{drag=null;});canvas.addEventListener('pointerleave',()=>{$('landscape-tooltip').hidden=true;});canvas.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();setPair(null,null);}});
    $('clear-landscape').addEventListener('click',()=>setPair(null,null));
    $('reset-landscape').addEventListener('click',()=>{landscapeYaw=-.6;landscapePitch=.9;renderLandscape();});
  }
  function redraw(){if(!data)return;Object.values(views).forEach(view=>view.draw());orbitViews.forEach(view=>view.draw());document.querySelector('.fusion-caption span:last-child').textContent=innerWidth<=760?'Swipe to explore · select a trajectory':'Hover or select a trajectory';renderSignatures();renderPair();renderFeatures();renderLayer();}
  async function init(){
    try{
      if(window.NDTL_DATA)data=window.NDTL_DATA;
      else{data={};for(const [key,file] of Object.entries({fusion:'fusion',classification:'classification',applications:'applications',features:'feature-analysis',landscape:'classification-landscape'})){const response=await fetch(`./data/${file}.json`);if(!response.ok)throw new Error(`${file}: ${response.status}`);data[key]=await response.json();}}
      pairLookup=new Map(data.landscape.pairs.map(pair=>[pairKey(pair.a,pair.b),pair]));systemLookup=new Map(data.landscape.systems.map(s=>[s.id,{...s,name:systemName(s.name)}]));
      for(const [name,id,color] of [['heroA','hero-parent-a',C.slate],['heroCreated','hero-created',C.gold],['heroB','hero-parent-b',C.lavender]])views[name]=new Trajectory($(id),color);
      const initial=caseById('lorenz-lv');views.heroA.set(initial.frames.at(-1).trajectory);views.heroCreated.set(initial.frames.find(f=>f.weight===.5).trajectory);views.heroB.set(initial.frames[0].trajectory);
      document.querySelectorAll('button[data-case]').forEach(button=>button.addEventListener('click',()=>setCase(button.dataset.case)));
      document.querySelector('.case-tabs').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;const buttons=[...document.querySelectorAll('button[data-case]')],index=buttons.indexOf(document.activeElement);if(index<0)return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;buttons[next].focus();setCase(buttons[next].dataset.case);});
      const options='<option value="">Choose a system</option>'+data.landscape.systems.map(system=>`<option value="${esc(system.id)}">${esc(systemName(system.name))}</option>`).join('');$('system-a').innerHTML=options;$('system-b').innerHTML=options;
      for(const id of ['system-a','system-b'])$(id).addEventListener('change',()=>{let a=$('system-a').value,b=$('system-b').value;if(a&&a===b){if(id==='system-a')b=null;else a=null;}setPair(a,b);});
      document.querySelectorAll('button[data-pair]').forEach(button=>button.addEventListener('click',()=>{const pair=data.landscape.pairs.find(pair=>pair.id===button.dataset.pair);setPair(pair.a,pair.b);}));
      $('cluster-legend').innerHTML=data.classification.clusters.map(cluster=>`<span><i class="dot" style="background:${clusterColors[(cluster.id-1)%7]}"></i>Cluster ${cluster.id}</span>`).join('');attachLandscape();
      $('reverse-field').addEventListener('click',()=>{reverseField=!reverseField;renderFeatures();});$('layer-metric').addEventListener('change',renderLayer);
      const fused=data.applications.encryption.benchmark.find(row=>row.id==='fusion_vec_6');$('encryption-metrics').innerHTML=`<span>Entropy<strong>${fmt(fused.cipher_entropy_mean,4)}</strong></span><span>NPCR<strong>${fmt(fused.npcr_mean,2)}%</strong></span><span>UACI<strong>${fmt(fused.uaci_mean,2)}%</strong></span>`;
      setCase('lorenz-lv');renderPair();renderFeatures();renderLayer();
      let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(redraw,130);});document.documentElement.dataset.ready='true';
    }catch(error){console.error(error);$('load-notice').hidden=false;$('load-notice').textContent='Research data could not be loaded. Rebuild data/research-data.js or use the supplied local preview server.';document.documentElement.dataset.ready='error';}
  }
  init();
})();
