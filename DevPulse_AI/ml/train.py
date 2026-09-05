from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
DF=pd.read_csv(ROOT/'ml/data/deployment_history.csv')
F=['files_changed','lines_added','lines_deleted','test_coverage','developer_experience','previous_incidents_30d','rollback_rate_90d','deployment_frequency_7d','database_changes','services_touched','hour_of_day']
X=DF[F];y=DF.incident_occurred
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
neg,pos=(yt==0).sum(),(yt==1).sum(); ratio=neg/max(pos,1)
mods={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'),'XGBoost':XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.045,subsample=.9,colsample_bytree=.9,objective='binary:logistic',eval_metric='logloss',random_state=42,scale_pos_weight=ratio)}
metrics={}
for n,m in mods.items():
 m.fit(Xt,yt);p=m.predict_proba(Xv)[:,1];pred=(p>=.5).astype(int)
 metrics[n]={'accuracy':round(accuracy_score(yv,pred),4),'precision':round(precision_score(yv,pred,zero_division=0),4),'recall':round(recall_score(yv,pred,zero_division=0),4),'f1':round(f1_score(yv,pred,zero_division=0),4),'roc_auc':round(roc_auc_score(yv,p),4)}
best=max(metrics,key=lambda n:metrics[n]['roc_auc'])
out=ROOT/'backend/models';out.mkdir(parents=True,exist_ok=True)
joblib.dump(mods[best],out/'xgboost_model.joblib');(out/'metrics.json').write_text(json.dumps({'selected_model':best,'models':metrics},indent=2));(out/'feature_columns.json').write_text(json.dumps(F,indent=2))
print('Selected:',best);print(json.dumps(metrics,indent=2))
