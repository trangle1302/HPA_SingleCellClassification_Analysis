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
    nu_area = []
    cell_area = []
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
            nu_area.append([n for n in nu_regions if n.label==cell_id][0].area)
            cell_area.append([n for n in cell_regions if n.label==cell_id][0].area)
    df['nucleus_area'] = nu_area
    df['cell_area'] = cell_area
    # Remove all cells without nucleus and cell areas (border cells)
    df = df[df.nucleus_area!=0]
    return df
    
def main():
    """
    d = '/data/kaggle-dataset/publicHPA_umap/results/webapp'
    if False:
        predictions0 = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_meta.csv')
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
    else:
        predictions0 = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21.csv')
        predictions0['cell_id']=[int(f.rsplit('_',1)[1]) for f in predictions0.id]
        predictions0['image_id']= predictions0.ID
        predictions = calculate_nuclei_size(predictions0, rm_border=True)
        predictions.to_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21_rm_border.csv', index=False)
    """
    
    d = '/data/kaggle-dataset/publicHPA_umap/results/webapp'
    cells_noborder = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv')
    cells_noborder['id'] = [f.split('_',1)[1] for f in cells_noborder.id]
    prediction2 = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21_individualthresholds_il.csv')
    prediction2['cell_id']=[int(f.rsplit('_',1)[1]) for f in prediction2.id]
    prediction2['image_id']= prediction2.ID
    predictions = prediction2[prediction2.id.isin(cells_noborder.id)]
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
    
    # Find multi-location SCV proteins in the same cell line
    proteins_tocheck = []
    proteins_tocheck2 = []
    topscv = multi_ensembl_ids
    scvproteins2 = []
    for p in topscv:
        predictions_p = predictions[predictions.ensembl_ids==p]#.groupby(['atlas_name', 'locations', 'target']).agg({'target':'count'})
        predictions_p = predictions_p[[len(str(f).split(','))>1 for f in predictions_p.locations_cleanedup]]
        if predictions_p.shape[0] == 0:
            continue
        cell_lines = set(predictions_p.atlas_name)
        scl = predictions_p.groupby(['locations_cleanedup', 'target']).agg({'target':'count'})
        agg_labels = list(set([f[1] for f in scl.index]))
        if ('Multi-Location' not in agg_labels) and (len(set(agg_labels)) >= 2):
                    proteins_tocheck.append(p)
        for c in cell_lines:
            il = set(list(set(predictions_p[predictions_p.atlas_name==c].locations_cleanedup))[0].split(','))
            if len(il) > 1:
                scl = predictions_p[predictions_p.atlas_name==c].groupby(['locations_cleanedup', 'target']).agg({'target':'count'})
                scl_noneg = predictions_p[(predictions_p.atlas_name==c) & (predictions_p.target!='Negative')].groupby(['locations_cleanedup', 'target']).agg({'target':'count'})
                agg_labels = list(set([f[1] for f in scl.index]))
                l  = [aggl for aggl in agg_labels if aggl not in ['Multi-Location', 'Negative']]
                if len(l) > 0: # & np.all(scl_noneg.target.values>1):
                    scvproteins2.append(p)            
                if ('Multi-Location' not in agg_labels) and (len(set(agg_labels)) >= 2):
                    proteins_tocheck2.append(p)

    print(len(set(proteins_tocheck)), set(proteins_tocheck))

    sc = proteins_tocheck2
    mapper = ifimages[['ensembl_ids','gene_names']].drop_duplicates()
    sc_gene_name = [mapper[mapper.ensembl_ids ==p].gene_names.values.tolist() for p in sc]
    sc_gene_name = [p[0] for p in sc_gene_name if len(p)>0]
    sc_gene_name = [p.split(',')[0] for p in sc_gene_name]

    # Gene set enrichment analysis       
    databases = ['GO_Biological_Process_2021', 'GO_Cellular_Component_2021', 'GO_Molecular_Function_2021', 'WikiPathway_2021_Human', 'KEGG_2021_Human']
    gl = sc_gene_name# list(set(proteins_tocheck2))
    # or a txt file path.
    enr = gseapy.enrichr(gene_list=gl, description='pathway', gene_sets=databases,
        outdir='test', background='hsapiens_gene_ensembl', cutoff=0.1, format='pdf')

    #check overlap with CCD proteins
    ccd = pd.read_csv('/home/trangle/Downloads/subcell_location_Cell.tsv', sep="\t")
    len(set(ccd.Ensembl).intersection(set(gl)))

if __name__ == '__main__':
    main()
