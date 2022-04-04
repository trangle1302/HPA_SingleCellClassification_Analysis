import os
import numpy as np
import pandas as pd
import json
import gseapy


label_names = {
    "0": "Nucleoplasm",
    "1": "Nuclear membrane",
    "2": "Nucleoli",
    "3": "Nucleoli fibrillar center",
    "4": "Nuclear speckles",
    "5": "Nuclear bodies",
    "6": "Endoplasmic reticulum",
    "7": "Golgi apparatus",
    "8": "Peroxisomes",
    "9": "Endosomes",
    "10": "Lysosomes",
    "11": "Intermediate filaments",
    "12": "Actin filaments",
    "13": "Focal adhesion sites",
    "14": "Microtubules",
    "15": "Microtubule ends",
    "16": "Cytokinetic bridge",
    "17": "Mitotic spindle",
    "18": "Microtubule organizing center",
    "19": "Centrosome",
    "20": "Lipid droplets",
    "21": "Plasma membrane",
    "22": "Cell junctions",
    "23": "Mitochondria",
    "24": "Aggresome",
    "25": "Cytosol",
    "26": "Cytoplasmic bodies",
    "27": "Rods & rings",
}

def calculate_volume(df_gene):
    x_range = df_gene.x.max() - df_gene.x.min()
    y_range = df_gene.y.max() - df_gene.y.min()
    z_range = df_gene.z.max() - df_gene.z.min() 
    volume = x_range*y_range*z_range
    return volume
    
    
def calculate_nuclei_size(df, rm_border = True):
    from skimage.measure import regionprops
    from imageio import imread
    from skimage.segmentation import clear_border
    import numpy as np
    
    mask_dir = "/data/kaggle-dataset/PUBLICHPA/mask/test"
    img_list = list(set(df.image_id))
    df['nucleus_area'] = 0
    df['cell_area'] = 0
    for image_id in img_list:
        nu_mask = imread(f"{mask_dir}/{image_id}_nucleimask.png")
        cell_mask = imread(f"{mask_dir}/{image_id}_cellmask.png")
        df_tmp = df[df.image_id == image_id]
        if rm_border: 
            nu_mask = clear_border(nu_mask)
            rm_cells = set(df_tmp.cell_id).difference(set(np.unique(nu_mask)))
            for rm_cell_id in rm_cells:
                cell_mask[cell_mask==rm_cell_id] = 0
        df_tmp = df_tmp[df_tmp.cell_id.isin(list(np.unique(nu_mask)))]
        nu_regions = regionprops(nu_mask)
        cell_regions = regionprops(cell_mask)
        for i,row in df_tmp.iterrows():
            cell_id = row.cell_id  
            df.loc[i,'nucleus_area'] = [n for n in nu_regions if n.label==cell_id][0].area
            df.loc[i,'cell_area'] = [n for n in cell_regions if n.label==cell_id][0].area
    # Remove all cells without nucleus and cell areas (border cells)
    df = df[df.nucleus_area!=0]
    return df
    
def main():
    d = '/data/kaggle-dataset/publicHPA_umap/results/webapp'
    predictions0 = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox.csv')
    predictions0['cell_id']=[int(f.rsplit('_',1)[1]) for f in predictions0.id]
    predictions0['image_id']=[f.rsplit('_',1)[0] for f in predictions0.id]
    predictions0['image_id']=[f.split('_',1)[1] for f in predictions0.image_id]
    predictions = calculate_nuclei_size(predictions0, rm_border=True)
    predictions.to_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv', index=False)

    rm_labels = ['Rods & rings', 'Cell junctions', 'Midbody', 'Midbody ring', 'Cytokinetic bridge']
    tmp = []
    for i,r in predictions.iterrows():
        ls = r.locations.split(',')
        ls = [l for l in ls if l not in rm_labels]
        if len(ls) > 1:
            tmp.append(",".join(ls))
        elif len(ls) == 1:
            tmp.append(ls[0])
        else:
            tmp.append("")
    predictions["locations_cleanedup"] = tmp

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
    multi = predictions[[len(str(f).split(','))>1 for f in predictions.locations_cleanedup]]
    multi_ensembl_ids = list(set(multi.ensembl_ids))
    predictions_sc['ensembl_ids'] = [f[0] for f in predictions_sc.ensembl_ids]
    # predictions_multi = predictions_sc[predictions_sc.ensembl_ids.isin(multi_ensembl_ids)]

proteins_tocheck = []
topscv = multi_ensembl_ids
scvproteins = []
for p in topscv:
    predictions_p = predictions[predictions.ensembl_ids==p]#.groupby(['atlas_name', 'locations', 'target']).agg({'target':'count'})
    predictions_p = predictions_p[[len(str(f).split(','))>1 for f in predictions_p.locations]]
    if predictions_p.shape[0] == 0:
        continue
    cell_lines = set(predictions_p.atlas_name)
    scl = predictions_p.groupby(['locations', 'target']).agg({'target':'count'})
    #agg_labels = list(set([f[1] for f in scl.index]))
    #if ('Multi-Location' not in agg_labels) and (len(set(agg_labels)) >= 2):
    #            proteins_tocheck.append(p)
    for c in cell_lines:
        il = set(list(set(predictions_p[predictions_p.atlas_name==c].locations))[0].split(','))
        if len(il) > 1:
            scl = predictions_p[predictions_p.atlas_name==c].groupby(['locations', 'target']).agg({'target':'count'})
            agg_labels = list(set([f[1] for f in scl.index]))
            l  = [aggl for aggl in agg_labels if aggl not in ['Multi-Location', 'Negative']]
            if len(l) > 0:
                scvproteins.append(p)            
            if ('Multi-Location' not in agg_labels) and (len(set(agg_labels)) >= 2):
                proteins_tocheck.append(p)

    print(len(set(proteins_tocheck)), set(proteins_tocheck))
    
    sc = proteins_tocheck
    mapper = ifimages[['ensembl_ids','gene_names']].drop_duplicates()
    sc_gene_name = [mapper[mapper.ensembl_ids ==p].gene_names.values.tolist() for p in sc]
    sc_gene_name = [p[0] for p in sc_gene_name]
    """
    gl = ['ANGPTL4', 'ICAM1', 'FCGBP', 'INF2', 'UPRT', 'RO60', 'GATD3A,GATD3B', 'BTN3A1', 'MGAT4C', 'GIGYF1', 'PPIB', 'SORBS1', 'OTUD7B', 'RNF2', 'GLIPR2', 'ACOT13', 'LMTK3', 'ALDH1L2', 'CA11', 
    'BCO1', 'NDUFAF1', 'CTNS', 'PHLDB1', 'RMDN1', 'COPS4', 'DEPP1', 'EIF2S2', 'POU6F1', 'VLDLR', 'CCDC137', 'EHD2', 'SPRY2', 'KIF23', 'TMEM259', 'UBE4A', 'ZBED6CL', 'SETX', 'DVL3', 'SMOC1', 
    'YJEFN3', 'MSH5', 'BAZ2A', 'SET,SETSIP', 'PTPRK', 'PRSS1,PRSS2,PRSS3', 'CIZ1', 'FKBPL', 'IP6K2', 'SH3GL2', 'VENTX', 'BIRC6', 'FEZF1', 'BTG4', 'MGAT5B', 'NLGN1', 'JAM2', 'LRRC26', 'CYTH4', 
    'CLECL1', 'SLC16A8', 'HMGXB4', 'NXPH3', 'TGFB1', 'AP003419.1,POLD4', 'CD226', 'FBP2', 'ACTR10', 'IL33', 'GJB7']
    sc_celline = []
    for p in sc:
        predictions_p = predictions[predictions.ensembl_ids==p].groupby(['atlas_name']).target.nunique().values.tolist()
        if len([e for e in predictions_p if e>2]) >0 :
            sc_celline.append(p)
    """
    # Gene set enrichment analysis
        
    databases = ['GO_Biological_Process_2013', 'GO_Biological_Process_2015', 'GO_Biological_Process_2017', 'GO_Biological_Process_2017b', 'GO_Biological_Process_2018', 
    'GO_Biological_Process_2021', 'GO_Cellular_Component_2013', 'GO_Cellular_Component_2015', 'GO_Cellular_Component_2017', 'GO_Cellular_Component_2017b', 'GO_Cellular_Component_2018', 
    'GO_Cellular_Component_2021', 'GO_Molecular_Function_2013', 'GO_Molecular_Function_2015', 'GO_Molecular_Function_2017', 'GO_Molecular_Function_2017b', 'GO_Molecular_Function_2018', 
    'GO_Molecular_Function_2021', 'WikiPathway_2021_Human', 'WikiPathways_2013', 'WikiPathways_2015', 'WikiPathways_2016', 'WikiPathways_2019_Human', 'KEGG_2016','KEGG_2019_Human','KEGG_2021_Human']
    gl = sc_gene_name

    # or a txt file path.
    gseapy.enrichr(gene_list=gl, description='pathway', gene_sets=databases,
                outdir='test', background='hsapiens_gene_ensembl', cutoff=0.05, format='pdf' )

    ccd = pd.read_csv('/home/trangle/Downloads/subcell_location_Cell.tsv', sep="\t")
    len(set(ccd.Ensembl).intersection(set(gl)))
    

if __name__ == '__main__':
    main()
