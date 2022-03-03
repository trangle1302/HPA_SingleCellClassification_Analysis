import os
import numpy as np
import pandas as pd
import json

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

    l_sc = [s for s in l if len(s)==1]
    len(set([item for sublist in l_sc for item in sublist])) # should be 11,833 genes for HPAv20 that is imaged with antibody that is not multi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     jj                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

    ifimages = pd.read_csv("/data/HPA-IF-images/IF-image.csv")
    ifimages = ifimages[ifimages.latest_version == 21.0]
    multi = ifimages[[len(str(f).split(','))>1 for f in ifimages.locations]]
    multi_ensembl_ids = list(set(multi.ensembl_ids))
    predictions_sc['ensembl_ids'] = [f[0] for f in predictions_sc.ensembl_ids]
    # predictions_multi = predictions_sc[predictions_sc.ensembl_ids.isin(multi_ensembl_ids)]

    proteins_tocheck = []
    topscv = multi_ensembl_ids
    predictions_sc = dict()
    for p in topscv:
        predictions_p = predictions[predictions.ensembl_ids==p].groupby(['atlas_name', 'target']).agg({'target':'count'})
        agg_labels = list(set([f[1] for f in predictions_p.index]))
        if 'Multi-Location' not in agg_labels and len(set(agg_labels)) > 2:
            proteins_tocheck += [p]
            predictions_sc[p] = dict(predictions_p)
    print(len(proteins_tocheck), proteins_tocheck)
    

    # Count number of ensemble_ids:
    l = [f.split(',') for f in meta.ensembl_ids]
    l = [item for sublist in l for item in sublist]
    len(set(l))

if __name__ == '__main__':
    main()