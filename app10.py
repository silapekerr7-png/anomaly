import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Arama sorgularında anomali tespiti :mag:")
dosya=st.file_uploader('Queries.csv dosyasını yükleyiniz:',type=['csv'])
oran=st.slider('Anomali oranı (%)',1,10,1)

if dosya:
    df=pd.read_csv(dosya)
    df['CTR']=df['CTR'].str.rstrip('%').astype('float')/100

    x=df[['Clicks','Impressions','CTR','Position']].copy()
    x['Clicks']=np.log1p(x['Clicks'])
    x['Impressions']=np.log1p(x['Impressions'])
    xs=StandardScaler().fit_transform(x)

    model=IsolationForest(n_estimators=200,contamination=oran/100,random_state=42)
    df['anomali']=model.fit_predict(xs)
    df['skor']=model.decision_function(xs)

    anomaliler=df[df['anomali']==-1].sort_values('skor')
    st.success(f'{len(df)} sorgudan {len(anomaliler)} tanesi anomali olarak bulundu.')
    st.dataframe(anomaliler[['Top queries','Clicks','Impressions','CTR','Position','skor']])

    st.subheader('Sıralama - CTR')
    df['durum']=df['anomali'].map({1:'Normal',-1:'Anomali'})
    st.scatter_chart(df,x='Position',y='CTR',color='durum')