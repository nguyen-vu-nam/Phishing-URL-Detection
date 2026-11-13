"""Three-model repair experiment. Select on validation before independent evaluation."""
from pathlib import Path
from zipfile import ZipFile
import io,json,time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from lightgbm import LGBMClassifier
from demo import normalized_host,extract
from features_v2 import lexical_features,fit_features,transform_features

ROOT=Path(__file__).resolve().parent
AUDIT=["https://dash.cloudflare.com/","https://github.com","https://translate.google.com/",
       "https://chatgpt.com/","https://support.microsoft.com/","https://www.wikipedia.org/",
       "https://docs.python.org/3/","https://www.mozilla.org/","https://www.bbc.com/",
       "https://www.example.org","https://www.example.org/"]

def prepare(frame):
    frame=frame.copy()
    frame["URL"]=frame["URL"].astype(str).str.strip()
    def host(url):
        try:
            return normalized_host(url)
        except (ValueError,UnicodeError):
            return None
    frame["host"]=frame["URL"].map(host)
    frame=frame.dropna(subset=["host","y"]).copy()
    frame["y"]=frame["y"].astype(int)
    groups={h:extract(h).top_domain_under_public_suffix or h for h in frame["host"].unique()}
    frame["group"]=frame["host"].map(groups)
    return frame

def metrics(y,prob):
    pred=prob>.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(f1=float(f1_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred)),fpr=float(fp/(tn+fp)),
                roc_auc=float(roc_auc_score(y,prob)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    with ZipFile(ROOT/"data/phishvn.zip") as z:
        vn=pd.read_csv(io.BytesIO(z.read("data/dataset_url.csv")),low_memory=False)
    vn=prepare(vn.loc[vn.tier.isin(["gold","silver"])].drop(columns="url").rename(columns={"url_norm":"URL"}).assign(
        y=lambda d:d.label.map({"benign":0,"phishing":1})))
    assert set(vn.y) == {0,1} and len(vn) > 18000
    audit=prepare(pd.DataFrame({"URL":AUDIT,"y":0}))
    blocked=set(vn.group)|set(audit.group)

    uci=pd.read_csv(ROOT/"data/PhiUSIIL_Phishing_URL_Dataset.csv",usecols=["URL","label"])
    uci=prepare(uci.assign(y=uci.label.eq(0).astype(int)))
    split=pd.read_csv(ROOT/"data/splits.csv",index_col="source_row")
    test_uci=uci.loc[split.index[split.split.eq("test")]]
    blocked |= set(test_uci.group)
    train_uci=uci.loc[split.index[split.split.eq("train")]]
    edge=pd.read_csv(ROOT/"data/edgephish.csv").rename(columns={"url":"URL"})
    edge=prepare(edge.assign(y=edge.label.map({"legitimate":0,"phishing":1})))
    legit=pd.read_csv(ROOT/"data/legitphish.csv")
    legit=prepare(legit.assign(y=legit.ClassLabel.map({0:1,1:0})))
    pieces=[]
    for name,frame,limit in [("PhiUSIIL",train_uci,20000),("EdgePhish",edge,45000),("LegitPhish",legit,45000)]:
        frame=frame.loc[~frame.group.isin(blocked)]
        sampled=pd.concat([g.sample(min(limit,len(g)),random_state=42) for _,g in frame.groupby("y")])
        pieces.append(sampled.assign(source=name))
    pool=pd.concat(pieces,ignore_index=True)
    conflicts=pool.groupby("URL").y.transform("nunique").gt(1)
    pool=pool.loc[~conflicts].drop_duplicates("URL").reset_index(drop=True)
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(pool,groups=pool.group))
    train,val=pool.iloc[a],pool.iloc[b]
    assert set(train.group).isdisjoint(val.group)
    assert set(pool.group).isdisjoint(blocked)
    counts={"train":len(train),"validation":len(val),"independent":len(vn),
            "train_sources":train.groupby(["source","y"]).size().to_string()}
    print(json.dumps(counts),flush=True)
    X,scaler,tfidf=fit_features(train,lexical_features(train))
    V=transform_features(val,lexical_features(val),scaler,tfidf)
    models={
        "Logistic Regression":LogisticRegression(solver="liblinear",C=1,max_iter=300,random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=100,max_depth=28,min_samples_leaf=2,n_jobs=6,random_state=42),
        "LightGBM":LGBMClassifier(n_estimators=250,num_leaves=47,min_child_samples=20,learning_rate=.08,
                                  colsample_bytree=.6,max_bin=63,n_jobs=6,random_state=42,verbosity=-1,force_col_wise=True),
    }
    validation={}
    for name,model in models.items():
        t=time.perf_counter()
        model.fit(X,train.y)
        validation[name]=metrics(val.y,model.predict_proba(V)[:,1])
        validation[name]["seconds"]=time.perf_counter()-t
        print(name,validation[name],flush=True)
    chosen=max(validation,key=lambda n:validation[n]["f1"])
    print("Selected before independent test:",chosen,flush=True)
    model=models[chosen]
    test={}
    for name,frame in [("PhishVN gold/silver",vn),("PhiUSIIL",test_uci)]:
        test[name]=metrics(frame.y,model.predict_proba(transform_features(frame,lexical_features(frame),scaler,tfidf))[:,1])
    audit_scores=dict(zip(audit.URL,map(float,model.predict_proba(
        transform_features(audit,lexical_features(audit),scaler,tfidf))[:,1]),strict=True))
    result=dict(counts=counts,validation=validation,selected=chosen,test=test,audit=audit_scores)
    joblib.dump((model,scaler,tfidf),ROOT/"models/model_repaired.joblib",compress=3)
    (ROOT/"evaluation_repaired.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":
    main()
