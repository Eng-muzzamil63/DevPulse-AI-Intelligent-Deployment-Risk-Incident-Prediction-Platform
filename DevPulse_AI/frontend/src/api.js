const API='http://localhost:8000/api';
async function req(p,o){const r=await fetch(API+p,{headers:{'Content-Type':'application/json'},...o});if(!r.ok)throw new Error(await r.text()||r.statusText);return r.json()}
export const getDashboard=()=>req('/dashboard');export const getDeployments=()=>req('/deployments');export const predictRisk=features=>req('/predict',{method:'POST',body:JSON.stringify({features})});
