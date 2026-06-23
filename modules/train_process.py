import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

DSP32_DIR="../dataset/dsp32"
N_MFCC=49
N_COEF=40

def load_group(path, name):
    data=np.fromfile(os.path.join(path,name+".bin"),dtype=np.float32)
    count=data.size//N_MFCC//N_COEF
    return data.reshape(count,N_MFCC,N_COEF)

def load_dataset(path):
    normal=load_group(path,"normal")

    fault=np.concatenate([
        load_group(path,"abnormal"),
        load_group(path,"d1"),
        load_group(path,"d2")
    ],axis=0)

    x=np.concatenate([normal,fault],axis=0)

    y=np.concatenate([
        np.zeros(len(normal),dtype=np.int32),
        np.ones(len(fault),dtype=np.int32)
    ])

    return x,y

def train():
    x,y=load_dataset(DSP32_DIR)
    x=x.reshape(len(x),-1)

    print("x shape =",x.shape)
    print("y shape =",y.shape)

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model=LogisticRegression(
        max_iter=5000,
        n_jobs=-1
    )

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    print("\nAccuracy =",accuracy_score(y_test,y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test,y_pred))

    print("\nClassification Report")
    print(classification_report(y_test,y_pred))

    prob=model.predict_proba(x_test)

    print("\nNormal Prob Mean =",prob[y_test==0][:,0].mean())
    print("Fault  Prob Mean =",prob[y_test==1][:,1].mean())

    return model


def eval_group(model,dsp32_dir,name):
    data=load_group(dsp32_dir,name)
    x=data.reshape(len(data),-1)
    p=model.predict_proba(x)[:,1]
    print(name,
          "mean=",p.mean(),
          "std=",p.std(),
          "max=",p.max())

if __name__=="__main__":
    model = train()
    eval_group(model,DSP32_DIR,"normal")
    eval_group(model,DSP32_DIR,"abnormal")
    eval_group(model,DSP32_DIR,"d1")
    eval_group(model,DSP32_DIR,"d2")
    eval_group(model,DSP32_DIR,"d3")
    eval_group(model,DSP32_DIR,"d4")