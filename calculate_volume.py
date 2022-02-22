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
    gene_dfs = predictions.groupby('gene_names').agg({'x': ['min','max'],
                                                       'y': ['min','max'],
                                                       'z': ['min','max']})
    gene_dfs['x_range'] = gene_dfs.x['max'] - gene_dfs.x['min']
    gene_dfs['y_range'] = gene_dfs.y['max'] - gene_dfs.y['min']
    gene_dfs['z_range'] = gene_dfs.z['max'] - gene_dfs.z['min']
    gene_dfs['volume'] = [np.float(r.x_range*r.y_range*r.z_range) for _,r in gene_dfs.iterrows()]
    gene_dfs = gene_dfs.sort_values(by=['volume'], ascending=False)
    gene_dfs.to_csv(f'{d}/gene_cell_spread_volume.csv', index=False)
    proteins_tocheck = []
    topscv = gene_dfs.index
    predictions_sc = dict()
    for p in topscv:
        tmp = predictions[predictions.gene_names==p].target.value_counts()
        if 'Multi-Location' not in tmp.index and len(tmp.index) > 2:
            print(p,tmp)
            proteins_tocheck += [p]
            predictions_sc[p] = dict(tmp)
    print(len(proteins_tocheck), proteins_tocheck)
    
    out_file = open(f'{d}/proteins_topscv.json', "w")
    json.dump(predictions_sc, out_file, indent = 6)
    out_file.close()

if __name__ == '__main__':
    main()