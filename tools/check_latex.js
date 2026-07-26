// Every $...$ / $$...$$ in every statement must parse with the real KaTeX.
// A formula that fails shows the reader raw source, which is worse than no
// formula, and it is invisible unless something checks. Run from the repo root:
//     node tools/check_latex.js
// Exits non-zero when anything fails, so it can gate a commit.
const fs=require('fs'), path=require('path');
// Resolve KaTeX relative to this script: an absolute path here worked on exactly
// one machine and broke every CI runner and every clone.
const ROOT=path.resolve(__dirname,'..');
const katex=require(path.join(ROOT,'editor','media','katex','katex.min.js'));
const RE=/\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g;
const TASKS=path.join(ROOT,'tasks');
const tasks=fs.readdirSync(TASKS).filter(d=>fs.existsSync(path.join(TASKS,d,'task.md')));
let spans=0, bad=0; const byTask={}, byErr={};
for(const d of tasks){
  const t=fs.readFileSync(path.join(TASKS,d,'task.md'),'utf8');
  let m;
  while((m=RE.exec(t))){
    const src=(m[1]!==undefined?m[1]:m[2]);
    const display=m[1]!==undefined;
    spans++;
    try{ katex.renderToString(src,{displayMode:display,throwOnError:true,strict:false}); }
    catch(e){
      bad++;
      (byTask[d]=byTask[d]||[]).push({src:src.trim().slice(0,90), err:e.message.split(' at position')[0].slice(0,80)});
      const k=e.message.replace(/'.*?'/g,"'X'").split(' at position')[0].slice(0,60);
      byErr[k]=(byErr[k]||0)+1;
    }
  }
}
console.log(`math spans: ${spans} | failing: ${bad} | tasks affected: ${Object.keys(byTask).length} / ${tasks.length}`);
console.log('\nerror classes:');
Object.entries(byErr).sort((a,b)=>b[1]-a[1]).slice(0,12).forEach(([k,v])=>console.log(`  ${String(v).padStart(4)}  ${k}`));
console.log('\nsamples:');
Object.entries(byTask).slice(0,6).forEach(([d,l])=>{console.log('  '+d); l.slice(0,2).forEach(x=>console.log(`      ${x.err}\n        ${x.src}`));});
if(bad) fs.writeFileSync(path.join(require('os').tmpdir(),'katex_bad.json'), JSON.stringify(byTask));
process.exit(bad ? 1 : 0);
