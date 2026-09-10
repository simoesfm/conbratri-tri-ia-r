import json, math, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.special import expit, logsumexp
from scipy.stats import norm, chi2, multivariate_normal
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SRC = Path("/workspace/scratch/e940aa526892/upload/matriz-PS-2024-6o-ano-matematica.xlsx")
OUT = Path("/workspace/scratch/e940aa526892/analysis_outputs_2024")
OUT.mkdir(exist_ok=True)

# Read the scored response matrix. Planilha2 has 20 items plus the total score.
raw = pd.read_excel(SRC, sheet_name="Planilha1", header=None)
# The last three rows are spreadsheet summaries (counts, item numbers and proportions),
# not candidates. Retain only rows with a recorded total and binary item responses.
candidate_mask = raw.iloc[:,20].notna() & raw.iloc[:,:20].isin([0,1]).all(axis=1)
raw_candidates = raw.loc[candidate_mask].copy()
X20 = raw_candidates.iloc[:, :20].astype(float).to_numpy()
valid_items = list(range(20))  # items 5 and 19 annulled
item_labels = np.array([i + 1 for i in valid_items])
X = X20[:, valid_items].astype(int)
N, J = X.shape

def alpha_binary(x):
    j = x.shape[1]
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return j/(j-1)*(1-item_var/total_var)

scores = X.sum(axis=1)
pval = X.mean(axis=0)
sdx = X.std(axis=0, ddof=1)
rit = []
for j in range(J):
    rest = scores - X[:, j]
    rit.append(np.corrcoef(X[:, j], rest)[0,1])
rit = np.array(rit)

# Gauss-Hermite points for a standard normal latent distribution.
gh_x, gh_w = hermgauss(41)
theta = gh_x * np.sqrt(2)
prior_w = gh_w / np.sqrt(np.pi)
log_prior = np.log(prior_w)

def probabilities(model, pars):
    if model == "1PL":
        a = np.repeat(pars[0], J); b = pars[1:]; c = np.zeros(J)
    elif model == "2PL":
        a, b = pars[:J], pars[J:2*J]; c = np.zeros(J)
    else:
        a, b, c = pars[:J], pars[J:2*J], pars[2*J:3*J]
    P = c[:,None] + (1-c[:,None])*expit(a[:,None]*(theta[None,:]-b[:,None]))
    return np.clip(P, 1e-8, 1-1e-8), a, b, c

def e_step(model, pars):
    P, a, b, c = probabilities(model, pars)
    logpat = X @ np.log(P) + (1-X) @ np.log(1-P)
    logjoint = logpat + log_prior
    logden = logsumexp(logjoint, axis=1)
    post = np.exp(logjoint-logden[:,None])
    return post, float(logden.sum()), P

def fit_irt(model, max_iter=250, tol=1e-5):
    init_b = np.clip(-norm.ppf(np.clip(pval, .01, .99)), -3, 3)
    init_a = np.clip(1.2*np.maximum(rit, .15), .35, 1.5)
    if model == "1PL": pars = np.r_[1.0, init_b]
    elif model == "2PL": pars = np.r_[init_a, init_b]
    else: pars = np.r_[init_a, init_b, np.repeat(.12, J)]
    prev = -np.inf
    for it in range(max_iter):
        post, ll, P = e_step(model, pars)
        R = post.sum(axis=0)
        S = X.T @ post
        if model == "1PL":
            def obj(z):
                a = z[0]; b=z[1:]
                pp = expit(a*(theta[None,:]-b[:,None])); pp=np.clip(pp,1e-9,1-1e-9)
                return -np.sum(S*np.log(pp)+(R[None,:]-S)*np.log(1-pp))
            res=minimize(obj, pars, method="L-BFGS-B", bounds=[(.15,4)]+[(-5,5)]*J,
                         options={"maxiter":300,"ftol":1e-10})
            new=res.x
        else:
            aa=[]; bb=[]; cc=[]
            for j in range(J):
                if model == "2PL":
                    z0=[pars[j],pars[J+j]]; bounds=[(.15,4),(-5,5)]
                    def obj(z):
                        pp=expit(z[0]*(theta-z[1])); pp=np.clip(pp,1e-9,1-1e-9)
                        return -np.sum(S[j]*np.log(pp)+(R-S[j])*np.log(1-pp))
                else:
                    z0=[pars[j],pars[J+j],pars[2*J+j]]; bounds=[(.15,4),(-5,5),(.001,.35)]
                    def obj(z):
                        pp=z[2]+(1-z[2])*expit(z[0]*(theta-z[1])); pp=np.clip(pp,1e-9,1-1e-9)
                        return -np.sum(S[j]*np.log(pp)+(R-S[j])*np.log(1-pp))
                res=minimize(obj,z0,method="L-BFGS-B",bounds=bounds,options={"maxiter":200,"ftol":1e-10})
                aa.append(res.x[0]); bb.append(res.x[1]);
                if model=="3PL": cc.append(res.x[2])
            new=np.r_[aa,bb] if model=="2PL" else np.r_[aa,bb,cc]
        post2, ll2, _ = e_step(model,new)
        if abs(ll2-prev) < tol*(1+abs(ll2)):
            pars=new; ll=ll2; break
        pars=new; prev=ll2
    post,ll,P=e_step(model,pars)
    k={"1PL":J+1,"2PL":2*J,"3PL":3*J}[model]
    return {"model":model,"pars":pars,"post":post,"ll":ll,"aic":-2*ll+2*k,
            "bic":-2*ll+k*np.log(N),"k":k,"iterations":it+1,"converged":it+1<max_iter}

fits={m:fit_irt(m) for m in ["1PL","2PL","3PL"]}
comparison=pd.DataFrame([{k:v[k] for k in ["model","ll","k","aic","bic","iterations","converged"]} for v in fits.values()])
comparison["delta_AIC"]=comparison.aic-comparison.aic.min()
comparison["delta_BIC"]=comparison.bic-comparison.bic.min()
best_bic=comparison.loc[comparison.bic.idxmin(),"model"]
best_aic=comparison.loc[comparison.aic.idxmin(),"model"]

model_params=[]
for m,ff in fits.items():
    pp=ff["pars"]
    if m=="1PL": aa=np.repeat(pp[0],J); bb=pp[1:]; cc=np.zeros(J)
    elif m=="2PL": aa=pp[:J]; bb=pp[J:2*J]; cc=np.zeros(J)
    else: aa=pp[:J]; bb=pp[J:2*J]; cc=pp[2*J:3*J]
    for it,av,bv,cv in zip(item_labels,aa,bb,cc): model_params.append((m,it,av,bv,cv))
model_params=pd.DataFrame(model_params,columns=["Modelo","Item","a","b","c"])

# The 2PL is the principal interpretive model; model comparison is retained for judgment.
fit=fits["2PL"]
pars=fit["pars"]; a=pars[:J]; b=pars[J:2*J]; c=np.zeros(J)
post=fit["post"]
eap=post@theta
eap_var=post@(theta**2)-eap**2
eap_se=np.sqrt(np.maximum(eap_var,0))

def discr_class(v):
    return "muito baixa" if v<.35 else "baixa" if v<.65 else "moderada" if v<1.35 else "alta" if v<1.70 else "muito alta"
def diff_class(v):
    return "muito fácil" if v<-2 else "fácil" if v<-1 else "mediana" if v<=1 else "difícil" if v<=2 else "muito difícil"

# Approximate grouped item fit based on EAP deciles.
def p2(theta_vals,j): return expit(a[j]*(theta_vals-b[j]))
bins=pd.qcut(eap,10,labels=False,duplicates="drop")
fit_rows=[]
for j in range(J):
    stat=0; groups=0
    for g in np.unique(bins):
        idx=bins==g; n=idx.sum(); obs=X[idx,j].sum(); exp=p2(eap[idx],j).sum()
        if n>0 and exp>.5 and n-exp>.5:
            stat+=(obs-exp)**2/exp+((n-obs)-(n-exp))**2/(n-exp); groups+=1
    df=max(groups-3,1); pv=chi2.sf(stat,df)
    fit_rows.append((stat,df,pv))
fit_rows=np.array(fit_rows)

# Local dependence: residual correlations after conditioning on EAP.
pred=np.column_stack([p2(eap,j) for j in range(J)])
resid=X-pred
q3=np.corrcoef(resid,rowvar=False)
np.fill_diagonal(q3,np.nan)
q3_pairs=[]
for i in range(J):
    for j in range(i+1,J): q3_pairs.append((item_labels[i],item_labels[j],q3[i,j],abs(q3[i,j])))
q3_df=pd.DataFrame(q3_pairs,columns=["Item_1","Item_2","Q3","Abs_Q3"]).sort_values("Abs_Q3",ascending=False)

# Tetrachoric correlation matrix and eigenvalues for dimensionality evidence.
def tetra_pair(x,y):
    p1=np.clip(x.mean(),.002,.998); p2=np.clip(y.mean(),.002,.998)
    t1=norm.ppf(p1); t2=norm.ppf(p2); p11=np.mean((x==1)&(y==1))
    lo=max(0,p1+p2-1)+1e-7; hi=min(p1,p2)-1e-7; target=np.clip(p11,lo,hi)
    def f(r): return multivariate_normal.cdf([t1,t2],mean=[0,0],cov=[[1,r],[r,1]])-target
    try: return brentq(f,-.98,.98)
    except Exception: return np.corrcoef(x,y)[0,1]
tet=np.eye(J)
for i in range(J):
    for j in range(i+1,J): tet[i,j]=tet[j,i]=tetra_pair(X[:,i],X[:,j])
eig=np.linalg.eigvalsh(tet)[::-1]

# Information curves.
grid=np.linspace(-4,4,321)
Pgrid=expit(a[:,None]*(grid[None,:]-b[:,None]))
item_info=(a[:,None]**2)*Pgrid*(1-Pgrid)
test_info=item_info.sum(axis=0)
test_se=1/np.sqrt(np.maximum(test_info,1e-12))

items=pd.DataFrame({
    "Item":item_labels,"N":N,"Proporcao_acertos":pval,"Correlacao_item_total_corrigida":rit,
    "a_discriminacao_2PL":a,"Classificacao_discriminacao":list(map(discr_class,a)),
    "b_dificuldade_2PL":b,"Classificacao_dificuldade":list(map(diff_class,b)),
    "Qui_quadrado_ajuste_aprox":fit_rows[:,0],"gl_ajuste":fit_rows[:,1].astype(int),"p_ajuste_aprox":fit_rows[:,2],
    "Informacao_maxima_item":item_info.max(axis=1),"Theta_informacao_maxima":grid[item_info.argmax(axis=1)]
})
flags=[]
for _,r in items.iterrows():
    f=[]
    if r.Correlacao_item_total_corrigida<.20:f.append("correlação item-total baixa")
    if r.a_discriminacao_2PL<.65:f.append("discriminação baixa")
    if abs(r.b_dificuldade_2PL)>2:f.append("dificuldade extrema")
    if r.p_ajuste_aprox<.01:f.append("possível desajuste")
    flags.append("; ".join(f) if f else "sem alerta principal")
items["Diagnostico"]=flags

persons=pd.DataFrame({"Candidato":[f"{i+1:03d}" for i in range(N)],"Escore_20_itens":scores,"Theta_EAP_2PL":eap,"Erro_padrao_EAP":eap_se})
curve=pd.DataFrame({"Theta":grid,"Informacao_teste":test_info,"Erro_padrao_teste":test_se})
for j,it in enumerate(item_labels): curve[f"Info_item_{it}"]=item_info[j]
dimens=pd.DataFrame({"Componente":np.arange(1,J+1),"Autovalor_tetracorico":eig,"Percentual":eig/eig.sum()})
summary={
    "N":N,"items_total":20,"items_valid":J,"items_annulled":[],"alpha":alpha_binary(X),
    "score_mean":float(scores.mean()),"score_sd":float(scores.std(ddof=1)),"score_min":int(scores.min()),"score_max":int(scores.max()),
    "first_eigen":float(eig[0]),"second_eigen":float(eig[1]),"eigen_ratio":float(eig[0]/eig[1]),
    "variance_first":float(eig[0]/eig.sum()),"best_aic":best_aic,"best_bic":best_bic,
    "q3_max_abs":float(q3_df.Abs_Q3.max()),"q3_pairs_over_020":int((q3_df.Abs_Q3>.20).sum()),
    "theta_info_ge_5_min":float(grid[test_info>=5].min()) if np.any(test_info>=5) else None,
    "theta_info_ge_5_max":float(grid[test_info>=5].max()) if np.any(test_info>=5) else None,
    "model_interpreted":"2PL"
}

# Save structured results.
items.to_csv(OUT/"itens.csv",index=False)
items.to_json(OUT/"itens.json",orient="records",force_ascii=False)
comparison.to_csv(OUT/"comparacao_modelos.csv",index=False)
comparison.to_json(OUT/"comparacao_modelos.json",orient="records",force_ascii=False)
model_params.to_csv(OUT/"parametros_modelos.csv",index=False)
model_params.to_json(OUT/"parametros_modelos.json",orient="records",force_ascii=False)
persons.to_csv(OUT/"proficiências.csv",index=False)
persons.to_json(OUT/"proficiencias.json",orient="records",force_ascii=False)
curve.to_csv(OUT/"curvas_informacao.csv",index=False)
curve.to_json(OUT/"curvas_informacao.json",orient="records",force_ascii=False)
q3_df.to_csv(OUT/"dependencia_local_q3.csv",index=False)
q3_df.to_json(OUT/"dependencia_local_q3.json",orient="records",force_ascii=False)
pd.DataFrame(q3,index=item_labels,columns=item_labels).to_csv(OUT/"matriz_q3.csv")
pd.DataFrame(tet,index=item_labels,columns=item_labels).to_csv(OUT/"matriz_tetracorica.csv")
dimens.to_csv(OUT/"dimensionalidade.csv",index=False)
dimens.to_json(OUT/"dimensionalidade.json",orient="records",force_ascii=False)
with open(OUT/"resumo.json","w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)

# Plots.
plt.style.use("seaborn-v0_8-whitegrid")
colors=plt.cm.tab20(np.linspace(0,1,J))
fig,ax=plt.subplots(figsize=(10,6))
for j,it in enumerate(item_labels): ax.plot(grid,expit(a[j]*(grid-b[j])),label=str(it),lw=1.6,color=colors[j])
ax.set(xlabel="Proficiência (θ)",ylabel="Probabilidade de acerto",title="Curvas características dos itens — modelo 2PL")
ax.legend(title="Item",ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(OUT/"curvas_caracteristicas.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5.5)); ax.plot(grid,test_info,color="#0B6E4F",lw=2.5); ax2=ax.twinx(); ax2.plot(grid,test_se,color="#B5651D",lw=2,ls="--")
ax.set(xlabel="Proficiência (θ)",ylabel="Informação do teste",title="Informação e erro de medida da prova — 2PL"); ax2.set_ylabel("Erro-padrão",color="#B5651D")
fig.tight_layout(); fig.savefig(OUT/"informacao_teste.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5.8)); sc=ax.scatter(b,a,c=pval,cmap="viridis",s=75,edgecolor="white")
for j,it in enumerate(item_labels): ax.annotate(str(it),(b[j],a[j]),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.axhline(.65,color="#B33A3A",ls="--",lw=1); ax.axvline(0,color="#777",lw=.8); ax.set(xlabel="Dificuldade b",ylabel="Discriminação a",title="Mapa dos itens — dificuldade e discriminação (2PL)")
fig.colorbar(sc,ax=ax,label="Proporção de acertos"); fig.tight_layout(); fig.savefig(OUT/"mapa_itens.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,5)); ax.bar(np.arange(1,J+1),eig,color=["#0B6E4F"]+["#A8C5B8"]*(J-1)); ax.axhline(1,color="#B5651D",ls="--"); ax.set(xlabel="Componente",ylabel="Autovalor",title="Autovalores da matriz tetracórica"); ax.set_xticks(np.arange(1,J+1)); fig.tight_layout(); fig.savefig(OUT/"scree_plot.png",dpi=180); plt.close(fig)

print(json.dumps(summary,ensure_ascii=False,indent=2))
print(items.sort_values(["Diagnostico","a_discriminacao_2PL"]).to_string(index=False))
print(comparison.to_string(index=False))
