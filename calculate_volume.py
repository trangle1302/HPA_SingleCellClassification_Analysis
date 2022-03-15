import os
import numpy as np
import pandas as pd
import json
import gseapy

def calculate_volume(df_gene):
    x_range = df_gene.x.max() - df_gene.x.min()
    y_range = df_gene.y.max() - df_gene.y.min()
    z_range = df_gene.z.max() - df_gene.z.min() 
    volume = x_range*y_range*z_range
    return volume
    
def main():
    d = '/data/kaggle-dataset/publicHPA_umap/results/webapp'
    predictions = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox.csv')
    gene_dfs = predictions.groupby(['ensembl_ids','gene_names','atlas_name']).agg({'x': ['min','max'],
                                                        'y': ['min','max'],
                                                        'z': ['min','max']})
    gene_dfs['x_range'] = gene_dfs.x['max'] - gene_dfs.x['min']
    gene_dfs['y_range'] = gene_dfs.y['max'] - gene_dfs.y['min']
    gene_dfs['z_range'] = gene_dfs.z['max'] - gene_dfs.z['min']
    gene_dfs['volume'] = [np.float(r.x_range*r.y_range*r.z_range) for _,r in gene_dfs.iterrows()]
    gene_dfs = gene_dfs.sort_values(by=['volume'], ascending=False)
    gene_dfs.to_csv(f'{d}/gene_cell_spread_volume.csv', index=False)

    predictions_sc = predictions.groupby(['ensembl_ids','gene_names','atlas_name','target']).agg(
        {
            'ensembl_ids':'unique',
            'target': 'count', 
            'gene_names':'unique', 
            'atlas_name':'unique'
        })
    l = [f[0].split(',') for f in predictions_sc.ensembl_ids]
    len(set([item for sublist in l for item in sublist])) # should be 12,770 for HPAv20

    l_sa = [s for s in l if len(s)==1]
    len(set([item for sublist in l_sa for item in sublist])) # should be 11,833 genes for HPAv20 that is imaged with antibody that is not multi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     jj                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

    ifimages = pd.read_csv("/data/HPA-IF-images/IF-image.csv")
    ifimages = ifimages[ifimages.latest_version == 21.0]
    multi = ifimages[[len(str(f).split(','))>1 for f in ifimages.locations]]
    multi_ensembl_ids = list(set(multi.ensembl_ids))
    predictions_sc['ensembl_ids'] = [f[0] for f in predictions_sc.ensembl_ids]
    # predictions_multi = predictions_sc[predictions_sc.ensembl_ids.isin(multi_ensembl_ids)]

proteins_tocheck = []
topscv = multi_ensembl_ids
scvproteins = []
for p in topscv[:10]:
    predictions_p = predictions[predictions.ensembl_ids==p].groupby(['atlas_name', 'target']).agg({'target':'count'})
    predictions_p
    agg_labels = list(set([f[1] for f in predictions_p.index]))
    l  = [aggl for aggl in agg_labels if aggl not in ['Multi-Location', 'Negative']]
    if len(l) > 0:
        scvproteins.append(p)
        if 'Multi-Location' not in agg_labels and len(set(agg_labels)) >= 2:
            proteins_tocheck.append(p)
print(len(proteins_tocheck), proteins_tocheck)
    

    # Count number of ensemble_ids:
    l = [f.split(',') for f in meta.ensembl_ids]
    l = [item for sublist in l for item in sublist]
    len(set(l))

    sc = proteins_tocheck
    mapper = ifimages[['ensembl_ids','gene_names']].drop_duplicates()
    sc_gene_name = [mapper[mapper.ensembl_ids ==p].gene_names.values.tolist() for p in sc]
    sc_gene_name = [p[0] for p in sc_gene_name]

    gl = ['ANGPTL4', 'ICAM1', 'FCGBP', 'INF2', 'UPRT', 'RO60', 'GATD3A,GATD3B', 'BTN3A1', 'MGAT4C', 'GIGYF1', 'PPIB', 'SORBS1', 'OTUD7B', 'RNF2', 'GLIPR2', 'ACOT13', 'LMTK3', 'ALDH1L2', 'CA11', 
    'BCO1', 'NDUFAF1', 'CTNS', 'PHLDB1', 'RMDN1', 'COPS4', 'DEPP1', 'EIF2S2', 'POU6F1', 'VLDLR', 'CCDC137', 'EHD2', 'SPRY2', 'KIF23', 'TMEM259', 'UBE4A', 'ZBED6CL', 'SETX', 'DVL3', 'SMOC1', 
    'YJEFN3', 'MSH5', 'BAZ2A', 'SET,SETSIP', 'PTPRK', 'PRSS1,PRSS2,PRSS3', 'CIZ1', 'FKBPL', 'IP6K2', 'SH3GL2', 'VENTX', 'BIRC6', 'FEZF1', 'BTG4', 'MGAT5B', 'NLGN1', 'JAM2', 'LRRC26', 'CYTH4', 
    'CLECL1', 'SLC16A8', 'HMGXB4', 'NXPH3', 'TGFB1', 'AP003419.1,POLD4', 'CD226', 'FBP2', 'ACTR10', 'IL33', 'GJB7']
    sc_celline = []
    for p in sc:
        predictions_p = predictions[predictions.ensembl_ids==p].groupby(['atlas_name']).target.nunique().values.tolist()
        if len([e for e in predictions_p if e>2]) >0 :
            sc_celline.append(p)
    # Gene set enrichment analysis
    expression_dataframe = pd.DataFrame()
    gl = ['SCARA3', 'LOC100044683', 'CMBL', 'CLIC6', 'IL13RA1', 'TACSTD2', 'DKKL1', 'CSF1',
     'SYNPO2L', 'TINAGL1', 'PTX3', 'BGN', 'HERC1', 'EFNA1', 'CIB2', 'PMP22', 'TMEM173']

    gseapy.enrichr(gene_list=gl, description='pathway', gene_sets='KEGG_2016', outdir='test')

    # or a txt file path.
    gseapy.enrichr(gene_list='gene_list.txt', description='pathway', gene_sets='KEGG_2016',
                outdir='test', cutoff=0.05, format='png' )

if __name__ == '__main__':
    main()
