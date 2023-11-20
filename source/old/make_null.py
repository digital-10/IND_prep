import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\jh\0py_dev\digitalship\data\higgs\higgs_ori.csv")
print(df['Label'].value_counts(True))
print(df.columns)
## 널 만들기
cols = ['DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
df1 = df.copy()
for i, col in enumerate(cols):
    df1.loc[df1.sample(frac=0.3).index, col] = np.nan
df1.to_csv(r"C:\Users\jh\0py_dev\digitalship\data\higgs\higgs.csv", index=False)