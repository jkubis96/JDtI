import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import os
import math   
from sklearn.decomposition import PCA
import plotly.express as px
from joblib import Parallel, delayed
from adjustText import adjust_text
import seaborn as sns
from harmony import harmonize
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch, FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull
from sklearn.cluster import MeanShift
import pickle
from sklearn.cluster import DBSCAN
import umap
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from scipy import sparse
from scipy.io import mmwrite
  
from .utils import *




class Clustering:
    
    """
    A class for performing dimensionality reduction, clustering, and visualization
    on high-dimensional data (e.g., single-cell gene expression).
 
    The class provides methods for:
    - Normalizing and extracting subsets of data
    - Principal Component Analysis (PCA) and related clustering
    - Uniform Manifold Approximation and Projection (UMAP) and clustering
    - Visualization of PCA and UMAP embeddings
    - Harmonization of batch effects
    - Accessing processed data and cluster labels
 
    Methods
    -------
    add_data_frame(data, metadata)
        Class method to create a Clustering instance from a DataFrame and metadata.
    get_partial_data(names=None, features=None, name_slot='cell_names')
        Return a subset of the data by sample names and/or features.
    harmonize_sets(harmonize_type='harmony')
        Perform batch effect harmonization on PCA data.
    perform_PCA(pc_num=0, width=8, height=6)
        Perform PCA on the dataset and visualize the first two PCs.
    knee_plot_PCA(width=8, height=6)
        Plot the cumulative variance explained by PCs to determine optimal dimensionality.
    find_clusters_PCA(pc_num=0, eps=0.5, min_samples=10, width=8, height=6, harmonized=False)
        Apply DBSCAN clustering to PCA embeddings and visualize results.
    perform_UMAP(factorize=False, umap_num=0, pc_num=0, harmonized=False, ...)
        Compute UMAP embeddings with optional parameter tuning.
    knee_plot_umap(eps=0.5, min_samples=10)
        Determine optimal UMAP dimensionality using silhouette scores.
    find_clusters_UMAP(umap_n=5, eps=0.5, min_samples=10, width=8, height=6)
        Apply DBSCAN clustering on UMAP embeddings and visualize clusters.
    UMAP_vis(names_slot='cell_names', set_sep=True, point_size=0.6, ...)
        Visualize UMAP embeddings with labels and optional cluster numbering.
    UMAP_feature(feature_name, features_data=None, point_size=0.6, ...)
        Plot a single feature over UMAP coordinates with customizable colormap.
    get_umap_data()
        Return the UMAP embeddings along with cluster labels if available.
    get_pca_data()
        Return the PCA results along with cluster labels if available.
    return_clusters(clusters='umap')
        Return the cluster labels for UMAP or PCA embeddings.
 
    Raises
    ------
    ValueError
        For invalid parameters, mismatched dimensions, or missing metadata.
    """
    
    def __init__(self, data, metadata):
        
        self.clustering_data = data
        
        if metadata is None:
            metadata = pd.DataFrame({'cell_names':list(data.columns)})
            
        self.clustering_metadata = metadata
        
        self.subclusters = None

    
    
    @classmethod
    def add_data_frame(cls, data:pd.DataFrame, metadata: pd.DataFrame | None):
        
        """
        Create a Clustering instance from a DataFrame and optional metadata.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data with features as rows and samples/cells as columns.
        metadata : pandas.DataFrame or None
            Optional metadata for the samples. 
            Each row corresponds to a sample/cell, and column names in this DataFrame 
            should match the sample/cell names in `data`. Columns can contain additional 
            information such as cell type, experimental condition, batch, sets, etc.
        
        Returns
        -------
        Clustering
            A new instance of the Clustering class.
        """
           
        return cls(data, metadata)
    
    
    def get_partial_data(self, names: list | str | None = None, 
                         features: list | str | None = None, 
                         name_slot: str = 'cell_names'):
        
        """
       Return a subset of the data filtered by sample names and/or feature names.
    
       Parameters
       ----------
       names : list, str, or None
           Names of samples to include. If None, all samples are considered.
       features : list, str, or None
           Names of features to include. If None, all features are considered.
       name_slot : str
           Column in metadata to use as sample names.
    
       Returns
       -------
       pandas.DataFrame
           Subset of the normalized data based on the specified names and features.
       """
   
        data = self.normalized_data.copy()
        
        if name_slot in self.input_metadata.columns:
            data.columns = self.input_metadata[name_slot]
        else:
            raise ValueError("'name_slot' not occured in data!'")
            
        if isinstance(features, str):
            features = [features]
        elif features is None:
            features = []
        
        if isinstance(names, str):
            names = [names]
        elif names is None:
            names = []
            
        features = [x.upper() for x in features]
        names = [x.upper() for x in names]
        
        columns_names = [x.upper() for x in data.columns]
        features_names = [x.upper() for x in data.index]

        columns_bool = [True if x in names else False for x in columns_names]
        features_bool = [True if x in features else False for x in features_names]

        if True not in columns_bool and True not in features_bool:
            raise ValueError("Nothing to return. Check 'names' and/or 'features'")
        
        if True in columns_bool:
            data = data.loc[:,columns_bool]
        
        if True in features_bool:
            data = data.loc[features_bool,:]
        
        return data
    
    
        
    def harmonize_sets(self, 
                       harmonize_type: str = 'harmony',
                       batch_col: str = 'sets'):
        
        """
        Perform batch effect harmonization on PCA embeddings.
    
        Parameters
        ----------
        harmonize_type : str, default 'harmony'
            Type of harmonization to apply (currently supports 'harmony').
        batch_col : str, default 'sets'
            Name of the column in `metadata` that contains batch information for the samples/cells.
        
        Returns
        -------
        None
            Updates the `harmonized_PCA` attribute with harmonized data.
        """
        
        data_mat = np.array(self.PCA)     
        
        metadata = self.input_metadata
    
        self.harmonized_PCA = pd.DataFrame(harmonize(data_mat, metadata, batch_key=batch_col))
        
        self.harmonized_PCA.columns = self.PCA.columns
        
            
    def perform_PCA(self, 
                    pc_num:int = 0,
                    width = 8, 
                    height = 6):
    
        """
        Perform Principal Component Analysis (PCA) on the dataset.
      
        This method standardizes the data, applies PCA, stores results as attributes,
        and generates a scatter plot of the first two principal components.
      
        Parameters
        ----------
        pc_num : int, default 0
            Number of principal components to compute.
            If 0, computes all available components.
        width : int or float, default 8
            Width of the PCA figure.
        height : int or float, default 6
            Height of the PCA figure.
      
        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot showing the first two principal components.
      
        Updates
        -------
        self.PCA : pandas.DataFrame
            DataFrame with principal component scores for each sample.
        self.explained_var : numpy.ndarray
            Percentage of variance explained by each principal component.
        self.cumulative_var : numpy.ndarray
            Cumulative explained variance.
        """
            
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.clustering_data.T)
        
        if pc_num == 0:
            pc_num = data_scaled.shape[1]
    
        
        pca = PCA(n_components=pc_num, random_state=42)  
        
        
        principal_components = pca.fit_transform(data_scaled)
        
        pca_df = pd.DataFrame(data=principal_components, columns=['PC' + str(x + 1) for x in range(pc_num)])
        
        self.explained_var = pca.explained_variance_ratio_ * 100  
        self.cumulative_var = np.cumsum(self.explained_var)
        
        self.PCA = pca_df
        
        fig = plt.figure(figsize=(width,height))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
        plt.show()
        
        return fig
        
    
    def knee_plot_PCA(self,
                      width:int = 8, 
                      height:int = 6):
        
        """
        Plot cumulative explained variance to determine the optimal number of PCs.
     
        Parameters
        ----------
        width : int, default 8
            Width of the figure.
        height : int or, default 6
            Height of the figure.
     
        Returns
        -------
        matplotlib.figure.Figure
            Line plot showing cumulative variance explained by each PC.
        """
   
        fig_knee = plt.figure(figsize=(width, height))
        plt.plot(range(1, len(self.explained_var) + 1), self.cumulative_var, marker='o')
        plt.xlabel('PC (n components)')
        plt.ylabel('Cumulative explained variance (%)')
        plt.grid(True)
        
        xticks = [1] + list(range(5, len(self.explained_var) + 1, 5))
        plt.xticks(xticks, rotation=60)

        plt.show()
        
        return fig_knee
        
        
    
    def find_clusters_PCA(self, 
                          pc_num:int = 0, 
                          eps:float = 0.5, 
                          min_samples:int = 10,
                          width:int = 8, 
                          height:int = 6,
                          harmonized:bool = False):
        
        """
        Apply DBSCAN clustering to PCA embeddings and visualize the results.
    
        This method performs density-based clustering (DBSCAN) on the PCA-reduced
        dataset. Cluster labels are stored in the object's metadata, and a scatter
        plot of the first two principal components with cluster annotations is returned.
    
        Parameters
        ----------
        pc_num : int, default 0
            Number of principal components to use for clustering.
            If 0, uses all available components.
        eps : float, default 0.5
            Maximum distance between two points for them to be considered
            as neighbors (DBSCAN parameter).
        min_samples : int, default 10
            Minimum number of samples required to form a cluster (DBSCAN parameter).
        width : int, default 8
            Width of the output scatter plot.
        height : int, default 6
            Height of the output scatter plot.
        harmonized : bool, default False
            If True, use harmonized PCA data (`self.harmonized_PCA`).
            If False, use standard PCA results (`self.PCA`).
    
        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot of the first two principal components colored by
            cluster assignments.
    
        Updates
        -------
        self.clustering_metadata['PCA_clusters'] : list
            Cluster labels assigned to each cell/sample.
        self.input_metadata['PCA_clusters'] : list, optional
            Cluster labels stored in input metadata (if available).
        """
    
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        
        if harmonized:
            PCA = self.harmonized_PCA
        else: 
            PCA = self.PCA
            
        dbscan_labels = dbscan.fit_predict(PCA)
        
        
        pca_df = pd.DataFrame(PCA)
        pca_df['Cluster'] = dbscan_labels  
        
        
        fig = plt.figure(figsize=(width, height))

        for cluster_id in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
            plt.scatter(
                cluster_data['PC1'],
                cluster_data['PC2'],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
    
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(
            title='Clusters',
            loc='center left',
            bbox_to_anchor=(1.0, 0.5)  
        )
        
        plt.grid(True)
        plt.show()
        

        self.clustering_metadata['PCA_clusters'] = list(dbscan_labels)
            
        try:
            self.input_metadata['PCA_clusters'] = list(dbscan_labels)
        except:
            pass
        
        return fig
        
    
       
    def perform_UMAP(self, 
                     factorize:bool = False,
                     umap_num:int = 0,
                     pc_num:int = 0,
                     harmonized:bool = False,
                     n_neighbors:int = 5,  
                     min_dist:float | int= 0.1, 
                     spread:float | int=1.0,              
                     set_op_mix_ratio:float | int =1.0,    
                     local_connectivity:int=1,    
                     repulsion_strength:float | int =1.0,  
                     negative_sample_rate:int=5,  
                     width:int = 8, 
                     height:int = 6):
    
        """
        Compute and visualize UMAP embeddings of the dataset.
      
        This method applies Uniform Manifold Approximation and Projection (UMAP)
        for dimensionality reduction on either raw, PCA, or harmonized PCA data.
        Results are stored as a DataFrame (`self.UMAP`) and a scatter plot figure
        (`self.UMAP_plot`).
      
        Parameters
        ----------
        factorize : bool, default False
            If True, categorical sample labels (from column names) are factorized
            and used as supervision in UMAP fitting.
        umap_num : int, default 0
            Number of UMAP dimensions to compute. If 0, matches the input dimension.
        pc_num : int, default 0
            Number of principal components to use as UMAP input.
            If 0, use all available components or raw data.
        harmonized : bool, default False
            If True, use harmonized PCA embeddings (`self.harmonized_PCA`).
            If False, use standard PCA or raw scaled data.
        n_neighbors : int, default 5
            UMAP parameter controlling the size of the local neighborhood.
        min_dist : float, default 0.1
            UMAP parameter controlling minimum allowed distance between embedded points.
        spread : int | float, default 1.0
            Effective scale of embedded space (UMAP parameter).
        set_op_mix_ratio : int | float, default 1.0
            Interpolation parameter between union and intersection in fuzzy sets.
        local_connectivity : int, default 1
            Number of nearest neighbors assumed for each point.
        repulsion_strength : int | float, default 1.0
            Weighting applied to negative samples during optimization.
        negative_sample_rate : int, default 5
            Number of negative samples per positive sample in optimization.
        width : int, default 8
            Width of the output scatter plot.
        height : int, default 6
            Height of the output scatter plot.
       
        
        Updates
        -------
        self.UMAP : pandas.DataFrame
            Table of UMAP embeddings with columns `UMAP1 ... UMAPn`.
        
      
        Notes
        -----
        - For supervised UMAP (`factorize=True`), categorical codes from column
          names of the dataset are used as labels.
        """

        scaler = StandardScaler()
        
        
        if pc_num == 0 and harmonized:
            data_scaled = self.harmonized_PCA
            
            
        elif pc_num == 0:
            data_scaled = scaler.fit_transform(self.clustering_data.T)


        else:
            if harmonized:
                
                data_scaled = self.harmonized_PCA.iloc[:,0:pc_num]
            else:

                data_scaled = self.PCA.iloc[:,0:pc_num]

             
        if umap_num == 0:
            
            umap_num = data_scaled.shape[1]

            reducer = umap.UMAP(
                                n_components=len(data_scaled.T),
                                random_state=42,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                spread=spread,
                                set_op_mix_ratio=set_op_mix_ratio,
                                local_connectivity=local_connectivity,
                                repulsion_strength=repulsion_strength,
                                negative_sample_rate=negative_sample_rate,
                                n_jobs=1  
                            )
             

        else:
            
            reducer = umap.UMAP(
                                n_components=umap_num,
                                random_state=42,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                spread=spread,
                                set_op_mix_ratio=set_op_mix_ratio,
                                local_connectivity=local_connectivity,
                                repulsion_strength=repulsion_strength,
                                negative_sample_rate=negative_sample_rate,
                                n_jobs=1  
                            )
             
        
        if factorize:
            embedding = reducer.fit_transform(X = data_scaled, 
                                              y = pd.Categorical(self.clustering_data.columns).codes)
        else:
            embedding = reducer.fit_transform(X = data_scaled)


        umap_df = pd.DataFrame(embedding, columns=['UMAP' + str(x + 1) for x in range(umap_num)])

        plt.figure(figsize=(width,height))
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.7)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True)
        
        plt.show()      

        self.UMAP = umap_df
        
        
    def knee_plot_umap(self, 
                       eps:int | float = 0.5, 
                       min_samples:int = 10):
       
        """
        Plot silhouette scores for different UMAP dimensions to determine optimal n_components.
    
        Parameters
        ----------
        eps : float, default 0.5
            DBSCAN eps parameter for clustering each UMAP dimension.
        min_samples : int, default 10
            Minimum number of samples to form a cluster in DBSCAN.
    
        Returns
        -------
        matplotlib.figure.Figure
            Silhouette score plot across UMAP dimensions.
        """
    
        umap_range=range(2, len(self.UMAP.T))
         
        silhouette_scores = []
 
        for n in umap_range:
           
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(np.array(self.UMAP)[:,:n])
         
            mask = labels != -1
            if len(set(labels[mask])) > 1: 
                score = silhouette_score(np.array(self.UMAP)[:,:n][mask], labels[mask])
            else:
                score = -1  
         
            silhouette_scores.append(score)
         
        fig = plt.figure(figsize=(10, 5))
        plt.plot(umap_range, silhouette_scores, marker='o')
        plt.xlabel('UMAP (n_components)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        return fig
               
        
 
     
    def find_clusters_UMAP(self, 
                           umap_n:int = 5, 
                           eps:float | float = 0.5, 
                           min_samples:int = 10,
                           width:int = 8, 
                           height:int = 6):
        
        """
        Apply DBSCAN clustering on UMAP embeddings and visualize clusters.
        
        This method performs density-based clustering (DBSCAN) on the UMAP-reduced
        dataset. Cluster labels are stored in the object's metadata, and a scatter
        plot of the first two UMAP components with cluster annotations is returned.
        
        Parameters
        ----------
        umap_n : int, default 5
            Number of UMAP dimensions to use for DBSCAN clustering.
            Must be <= number of columns in `self.UMAP`.
        eps : float | int, default 0.5
            Maximum neighborhood distance between two samples for them to be considered
            as in the same cluster (DBSCAN parameter).
        min_samples : int, default 10
            Minimum number of samples in a neighborhood to form a cluster (DBSCAN parameter).
        width : int, default 8
            Figure width.
        height : int, default 6
            Figure height.
     
        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot of the first two UMAP components colored by
            cluster assignments.
    
        Updates
        -------
        self.clustering_metadata['UMAP_clusters'] : list
            Cluster labels assigned to each cell/sample.
        self.input_metadata['UMAP_clusters'] : list, optional
            Cluster labels stored in input metadata (if available).
        """
   
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(np.array(self.UMAP)[:,:umap_n])

        umap_df = self.UMAP
        umap_df['Cluster'] = dbscan_labels  

        fig = plt.figure(figsize=(width,height))


        for cluster_id in sorted(umap_df['Cluster'].unique()):
            cluster_data = umap_df[umap_df['Cluster'] == cluster_id]
            plt.scatter(
                cluster_data['UMAP1'],
                cluster_data['UMAP2'],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
    
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(
            title='Clusters',
            loc='center left',
            bbox_to_anchor=(1.0, 0.5)  
        )
        plt.grid(True)
        
        
        self.clustering_metadata['UMAP_clusters'] = list(dbscan_labels)
            
        try:
            self.input_metadata['UMAP_clusters'] = list(dbscan_labels)
        except:
            pass
            
        
        return fig

        
    def UMAP_vis(self, 
                 names_slot: str = 'cell_names', 
                 set_sep: bool = True,
                 point_size: int | float = 0.6,
                 font_size: int | float = 6,
                 legend_split_col: int = 2,
                 width: int = 8,
                 height: int = 6,
                 inc_num: bool = True):
        
        
        """
        Visualize UMAP embeddings with sample labels based on specyfic metadata slot.
    
        Parameters
        ----------
        names_slot : str, default 'cell_names'
            Column in metadata to use as sample labels.
        set_sep : bool, default True
            If True, separate points by dataset.
        point_size : float, default 0.6
            Size of scatter points.
        font_size : int, default 6
            Font size for numbers on points.
        legend_split_col : int, default 2
            Number of columns in legend.
        width : int, default 8
            Figure width.
        height : int, default 6
            Figure height.
        inc_num : bool, default True
            If True, annotate points with numeric labels.
    
        Returns
        -------
        matplotlib.figure.Figure
            UMAP scatter plot figure.
        """
       
        umap_df = self.UMAP.iloc[:,0:2]
        umap_df['names'] = self.input_metadata[names_slot]  
        
        umap_df['names'] = self.input_metadata[names_slot]  
        
        if set_sep:
            
            if 'sets' in self.input_metadata.columns:
                umap_df['dataset'] = self.input_metadata['sets']  
            else:
                umap_df['dataset'] = 'default'
        
        else:
            umap_df['dataset'] = 'default'

        
        umap_df['tmp_nam'] =  umap_df['names'] + umap_df['dataset']
        
        umap_df["count"] = umap_df["tmp_nam"].map(umap_df["tmp_nam"].value_counts())
        
        
        
        numeric_df = pd.DataFrame(umap_df[['count', 'tmp_nam', 'names']].copy()).drop_duplicates().sort_values('count', ascending = False)
        numeric_df['numeric_values'] = range(0,numeric_df.shape[0])
        
        umap_df = umap_df.merge(numeric_df[['tmp_nam', 'numeric_values']], on='tmp_nam', how='left')

        
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        markers = ['o', 's', '^', 'D', 'P', '*', 'X']
        marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(umap_df['dataset'].unique())}
        
        
        cord_list = []
        
        for num, nam in zip(numeric_df['numeric_values'], numeric_df['names']):
    
            cluster_data = umap_df[umap_df['numeric_values'] == num]
           
            ax.scatter(
                cluster_data['UMAP1'],
                cluster_data['UMAP2'],
                label=f'{num} - {nam}',  
                marker=marker_map[cluster_data['dataset'].iloc[0]],
                alpha=0.6,
                s = point_size
                
            )
        
            
            coords = cluster_data[["UMAP1", "UMAP2"]].values
            
            dists = pairwise_distances(coords)
            
            sum_dists = dists.sum(axis=1)
            
            center_idx = np.argmin(sum_dists)
            center_point = coords[center_idx]
            
            cord_list.append(center_point)
            
        if inc_num:
            texts = []
            for (x, y), num in zip(cord_list, numeric_df['numeric_values']):
                texts.append(ax.text(x, y, str(num), ha='center', va='center',
                                     fontsize=font_size, color='black'))
            
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        ax.legend(
            title='Clusters',
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            ncol=legend_split_col,
            markerscale=5
        )
        
       
        ax.grid(True)
                
        plt.tight_layout()
                
    
        return fig
    
    
    def UMAP_feature(self, 
                 feature_name: str,
                 features_data: pd.DataFrame | None,
                 point_size: int | float = 0.6,
                 font_size: int | float = 6,
                 width: int = 8,
                 height: int = 6,
                 palette = 'light'):
        
        """
        Visualize UMAP embedding with expression levels of a selected feature.
    
        Each point (cell) in the UMAP plot is colored according to the expression
        value of the chosen feature, enabling interpretation of spatial patterns
        of gene activity or metadata distribution in low-dimensional space.
    
        Parameters
        ----------
        feature_name : str
           Name of the feature to plot.
        features_data : pandas.DataFrame or None, default None
            If None, the function uses the DataFrame containing the clustering data.
            To plot features not used in clustering, provide a wider DataFrame 
            containing the original feature values.
        point_size : float, default 0.6
            Size of scatter points in the plot.
        font_size : int, default 6
            Font size for axis labels and annotations.
        width : int, default 8
            Width of the matplotlib figure.
        height : int, default 6
            Height of the matplotlib figure.
        palette : str, default 'light'
            Color palette for expression visualization. Options are:
            - 'light'
            - 'dark'
            - 'green'
            - 'gray'

        Returns
        -------
        matplotlib.figure.Figure
            UMAP scatter plot colored by feature values.
        """
        
 
        umap_df = self.UMAP.iloc[:,0:2]

        if features_data is None:
            
            features_data = self.clustering_data
            
        if features_data.shape[1] != umap_df.shape[0]:
            raise ValueError("Imputed 'features_data' shape does not match the number of UMAP cells")    
            
        blist = [True if x.upper() == feature_name.upper() else False for x in features_data.index]
        
        if not any(blist):
            raise ValueError("Imputed feature_name is not included in the data")

            
        umap_df['feature'] = features_data.loc[blist, :].apply(lambda row: row.tolist(), axis=1).values[0]
        
     
        
        umap_df =  umap_df.sort_values('feature', ascending = True)
        
        
        import matplotlib.colors as mcolors
     
          
        if palette == 'light':
            palette = px.colors.sequential.Sunsetdark  

        elif palette == 'dark':
            palette = px.colors.sequential.thermal  

        elif palette == 'green':
            palette = px.colors.sequential.Aggrnyl  
            
        elif palette == 'gray':
            palette = px.colors.sequential.gray  
            palette = palette[::-1]


        else:
            raise ValueError('Palette not found. Use: "light", "dark", "gray", or "green"')


        
        converted = []
        for c in palette:
            rgb_255 = px.colors.unlabel_rgb(c)  
            rgb_01 = tuple(v/255. for v in rgb_255)
            converted.append(rgb_01)
        
     
        my_cmap = mcolors.ListedColormap(converted, name="custom")


            
        
        fig, ax = plt.subplots(figsize=(width, height))
        sc = ax.scatter(
            umap_df['UMAP1'],
            umap_df['UMAP2'],
            c=umap_df['feature'],
            s=point_size,
            cmap=my_cmap,  
            alpha=1.0,
            edgecolors='black',
            linewidths=0.1
        )
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f"{feature_name}")
        
       

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        
        ax.grid(True)
                
        plt.tight_layout()

        return fig
        
        
    def get_umap_data(self):
        
        """
        Retrieve UMAP embedding data with optional cluster labels.
      
        Returns the UMAP coordinates stored in `self.UMAP`. If clustering
        metadata is available (specifically `UMAP_clusters`), the corresponding
        cluster assignments are appended as an additional column.
      
        Returns
        -------
        pandas.DataFrame
            DataFrame containing UMAP coordinates (columns: 'UMAP1', 'UMAP2', ...).
            If available, includes an extra column 'clusters' with cluster labels.
      
        Notes
        -----
        - UMAP embeddings must be computed beforehand (e.g., using `perform_UMAP`).
        - Cluster labels are added only if present in `self.clustering_metadata`.
        """
     
        umap_data = self.UMAP
        
        try:
            umap_data['clusters'] = self.clustering_metadata['UMAP_clusters']
        except:
            pass
    
        return umap_data
    
    
    def get_pca_data(self):
        
        """
        Retrieve PCA embedding data with optional cluster labels.
     
        Returns the principal component scores stored in `self.PCA`. If clustering
        metadata is available (specifically `PCA_clusters`), the corresponding
        cluster assignments are appended as an additional column.
     
        Returns
        -------
        pandas.DataFrame
            DataFrame containing PCA coordinates (columns: 'PC1', 'PC2', ...).
            If available, includes an extra column 'clusters' with cluster labels.
     
        Notes
        -----
        - PCA must be computed beforehand (e.g., using `perform_PCA`).
        - Cluster labels are added only if present in `self.clustering_metadata`.
        """
   
        pca_data = self.PCA
        
         
        try:
            pca_data['clusters'] = self.clustering_metadata['PCA_clusters']
        except:
            pass
        
        return pca_data
        
    def return_clusters(self, clusters = 'umap'):
        
        """
        Retrieve cluster labels from UMAP or PCA clustering results.
    
        Parameters
        ----------
        clusters : str, default 'umap'
            Source of cluster labels to return. Must be one of:
            - 'umap': return cluster labels from UMAP embeddings.
            - 'pca' : return cluster labels from PCA embeddings.
    
        Returns
        -------
        list
            Cluster labels corresponding to the selected embedding method.
    
        Raises
        ------
        ValueError
            If `clusters` is not 'umap' or 'pca'.
    
        Notes
        -----
        - Requires that clustering has already been performed
          (e.g., using `find_clusters_UMAP` or `find_clusters_PCA`).
        """
        
        if clusters.lower() == 'umap':
            clusters_vector = self.clustering_metadata['UMAP_clusters']
        elif clusters.lower() == 'pca':
            clusters_vector = self.clustering_metadata['PCA_clusters']
        else:
            raise ValueError("Parameter 'clusters' must be either 'umap' or 'pca'.")

        return clusters_vector 

      

    

class COMPsc(Clustering):
    
    """
    A class `COMPsc` (Comparison of single-cell data) designed for the integration, 
    analysis, and visualization of single-cell datasets.  
   The class supports independent dataset integration, subclustering of existing clusters,
    marker detection, and multiple visualization strategies.  

    The COMPsc class provides methods for:

        - Normalizing and filtering single-cell data
        - Loading and saving sparse 10x-style datasets
        - Computing differential expression and marker genes
        - Clustering and subclustering analysis
        - Visualizing similarity and spatial relationships
        - Aggregating data by cell and set annotations
        - Managing metadata and renaming labels
        - Plotting gene detection histograms and feature scatters
 
    Methods
    -------
    project_dir(path_to_directory, project_list)
        Scans a directory to create a COMPsc instance mapping project names to their paths.
    save_project(name, path=os.getcwd())
        Saves the COMPsc object to a pickle file on disk.
    load_project(path)
        Loads a previously saved COMPsc object from a pickle file.
    reduce(reg, inc_set=False)
        Removes columns from data tables where column names contain a specified substring.
    get_data(set_info=False)
        Returns normalized data with optional set annotations in column names.
    get_metadata()
        Returns the stored input metadata.
    gene_calc()
        Calculates and stores per-cell gene detection counts as a pandas Series.
    gene_histograme(bins=100)
        Plots a histogram of genes detected per cell with an overlaid normal distribution.
    gene_threshold(min_n=None, max_n=None)
        Filters cells based on minimum and/or maximum gene detection thresholds.
    load_sparse_from_projects(normalized_data=False)
        Loads and concatenates sparse 10x-style datasets from project paths into count or normalized data.
    rename_names(mapping, slot='cell_names')
        Renames entries in a specified metadata column using a provided mapping dictionary.
    rename_subclusters(mapping)
        Renames subcluster labels using a provided mapping dictionary.
    save_sparse(path_to_save=os.getcwd(), name_slot='cell_names', data_slot='normalized')
        Exports data as 10x-compatible sparse files (matrix.mtx, barcodes.tsv, genes.tsv).
    normalize_data(normalize=True, normalize_factor=100000)
        Normalizes raw counts to counts-per-specified factor (e.g., CPM-like).
    statistic(cells=None, sets=None, min_exp=0.01, min_pct=0.1, n_proc=10)
        Computes per-feature differential expression statistics (Mann-Whitney U) comparing target vs. rest groups.
    calculate_difference_markers(min_exp=0, min_pct=0.25, n_proc=10, force=False)
        Computes and caches differential markers using the statistic method.
    clustering_features(features_list=None, name_slot='cell_names', p_val=0.05, top_n=25, adj_mean=True, beta=0.4)
        Prepares clustering input by selecting marker features and optionally smoothing cell values.
    average()
        Aggregates normalized data by averaging across (cell_name, set) pairs.
    estimating_similarity(method='pearson', p_val=0.05, top_n=25)
        Computes pairwise correlation and Euclidean distance between aggregated samples.
    similarity_plot(split_sets=True, set_info=True, cmap='seismic', width=12, height=10)
        Visualizes pairwise similarity as a scatter plot with correlation as hue and scaled distance as point size.
    spatial_similarity(set_info=True, bandwidth=1, n_neighbors=5, min_dist=0.1, legend_split=2, point_size=20, ...)
        Creates a UMAP-like visualization of similarity relationships with cluster hulls and nearest-neighbor arrows.
    subcluster_prepare(features, cluster)
        Initializes a Clustering object for subcluster analysis on a selected parent cluster.
    define_subclusters(umap_num=2, eps=0.5, min_samples=10, bandwidth=1, n_neighbors=5, min_dist=0.1, ...)
        Performs UMAP and DBSCAN clustering on prepared subcluster data and stores cluster labels.
    subcluster_features_scatter(colors='viridis', hclust='complete', img_width=3, img_high=5, label_size=6, ...)
        Visualizes averaged expression and occurrence of features for subclusters as a scatter plot.
    subcluster_DEG_scatter(top_n=3, min_exp=0, min_pct=0.25, p_val=0.05, colors='viridis', ...)
        Plots top differential features for subclusters as a features-scatter visualization.
    accept_subclusters()
        Commits subcluster labels to main metadata by renaming cell names and clears subcluster data.
     

    Raises
    ------
    ValueError
        For invalid parameters, mismatched dimensions, or missing metadata.
        
    """
    
    def __init__(self, objects = None, 
                
                ):
        
        self.objects = objects
        self.input_data = None
        self.input_metadata = None
        self.normalized_data = None
        self.agg_metadata = None
        self.agg_normalized_data = None
        self.similarity = None
        self.var_data = None
        self.subclusters_ = None
        
        
    @classmethod
    def project_dir(cls, path_to_directory, project_list):
        
        """
        Scan a directory and build a COMPsc instance mapping provided project names
        to their paths.
        
        Parameters
        ----------
        path_to_directory : str
            Path containing project subfolders.
        project_list : list[str]
            List of filenames (folder names) to include in the returned object map.
        
        Returns
        -------
        COMPsc
            New COMPsc instance with `objects` populated.
        
        Raises
        ------
        Exception
            A generic exception is caught and a message printed if scanning fails.
        
        Notes
        -----
        - Function attempts to match entries in `project_list` to directory
          names and constructs a simplified object key from the folder name.
        """
        try:
            objects = {}
            for filename in tqdm(os.listdir(path_to_directory)):
                for c in project_list:
                    if c == filename and os.path.isdir(filename):
                        f = os.path.join(path_to_directory, filename)
                        objects[str(re.sub('_count', '', re.sub('_fq','', re.sub(re.sub('_.*', '', filename) + '_', '', filename))))] = f
        
            return cls(objects)
        
        except:
            print("Something went wrong. Check the function input data and try again!")
    
    
    



    def save_project(self, name, path:str = os.getcwd()):
        
        """
        Save the COMPsc object to disk using pickle.

        Parameters
        ----------
        name : str
            Base filename (without extension) to use when saving.
        path : str, default os.getcwd()
            Directory in which to save the project file.

        Returns
        -------
        None

        Side Effects
        ------------
        - Writes a file `<path>/<name>.jpkl` containing the pickled object.
        - Prints a confirmation message with saved path.
        """
        
        full = os.path.join(path, f"{name}.jpkl")

        with open(full, "wb") as f:
            pickle.dump(self, f)
            
        print(f"Project saved as {full}")


    @classmethod
    def load_project(cls, path):
        
        """
        Load a previously saved COMPsc project from a pickle file.

        Parameters
        ----------
        path : str
            Full path to the pickled project file.

        Returns
        -------
        COMPsc
            The unpickled COMPsc object.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist!")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    

    
    def reduce(self, reg:str, inc_set:bool = False):
        
        """
        Remove columns (cells) whose names contain a substring `reg` from available tables.

        Parameters
        ----------
        reg : str
            Substring to search for in column/cell names; matching columns will be removed.
        inc_set : bool, default False
            If True, column names are interpreted as 'cell_name # set' when matching.


        Update
        ------------
            Mutates `self.input_data`, `self.normalized_data`, `self.input_metadata`,
            `self.agg_normalized_data`, and `self.agg_metadata` (if they exist),
            removing columns/rows that match `reg`.
          
        Raises
        ------
            Raises ValueError if nothing matches the reduction mask.
        """
        
        if self.input_data != None:
            
            if inc_set:
                
                self.input_data.columns = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']
                
            else:
                
                self.input_data.columns = self.input_metadata['cell_names'] 
            
            mask = [reg not in x for x in self.input_data.columns]
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing found to reduce")
            
            self.input_data = self.input_data.loc[:, mask]

                
        
        if None is not self.normalized_data:
            
            if inc_set:
                
                self.normalized_data.columns = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']
                
                
            else:
                
                self.normalized_data.columns = self.input_metadata['cell_names'] 
                
                
            mask = [reg not in x for x in self.normalized_data.columns]
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing found to reduce")
            
            self.normalized_data = self.normalized_data.loc[:, mask]
            

        
        if None is not self.input_metadata:
            
            if inc_set:
                
                self.input_metadata['drop'] = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']
                
            else:
                
                self.input_metadata['drop'] = self.input_metadata['cell_names'] 
                
            
            mask = [reg not in x for x in self.input_metadata['drop']]
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing found to reduce")
            
            self.input_metadata = self.input_metadata.loc[mask, :].reset_index(drop = True)
            
            self.input_metadata = self.input_metadata.drop(columns=["drop"], errors="ignore")

            
        
        if None is not self.agg_normalized_data:
            
            if inc_set:
                
                self.agg_normalized_data.columns = self.agg_metadata['cell_names'] + ' # ' + self.agg_metadata['sets']
                
            else:
                
                self.agg_normalized_data.columns = self.agg_metadata['cell_names'] 
            
            
            mask = [reg not in x for x in self.agg_normalized_data.columns]
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing found to reduce")
            
            self.agg_normalized_data = self.agg_normalized_data.loc[:, mask]
            


        if None is not self.agg_metadata:
            
            if inc_set:
                
                self.agg_metadata['drop'] = self.agg_metadata['cell_names'] + ' # ' + self.agg_metadata['sets']
                
            else:
                
                self.agg_metadata['drop'] = self.agg_metadata['cell_names'] 
                
            
            mask = [reg not in x for x in self.agg_metadata['drop']]
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing found to reduce")
            
            self.agg_metadata = self.agg_metadata.loc[mask, :].reset_index(drop = True)
            
            self.agg_metadata = self.agg_metadata.drop(columns=["drop"], errors="ignore")
            
            
    
    def get_data(self, set_info:bool = False):
        
        """
        Return normalized data with optional set annotation appended to column names.

        Parameters
        ----------
        set_info : bool, default False
            If True, column names are returned as "cell_name # set"; otherwise
            only the `cell_name` is used.

        Returns
        -------
        pandas.DataFrame
            The `self.normalized_data` table with columns renamed according to `set_info`.

        Raises
        ------
        AttributeError
            If `self.normalized_data` or `self.input_metadata` is missing.
        """
        
        to_return = self.normalized_data

        if set_info:
            to_return.columns = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']
        else:
            to_return.columns = self.input_metadata['cell_names']

        return to_return
    
    
    def get_metadata(self):
        
        """
        Return the stored input metadata.

        Returns
        -------
        pandas.DataFrame
            `self.input_metadata` (may be None if not set).
        """
        
        to_return = self.input_metadata

        return to_return
    
    
    
    def gene_calc(self):
        
        """
        Calculate and store per-cell counts (e.g., number of detected genes).

        The method computes a binary (presence/absence) per cell and sums across
        features to produce `self.gene_calc`.


        Update
        ------
            Sets `self.gene_calc` as a pandas.Series.
        
        Side Effects
        ------------
        - Uses `self.input_data` when available, otherwise `self.normalized_data`.
        """
        
        if self.input_data != None:
            
           
            bin_col = self.input_data.columns.copy() 
            
            bin_col = bin_col.where(bin_col <= 0, 1) 
            
            sum_data = bin_col.sum(axis=0)
            
            self.gene_calc = sum_data

           
        elif None is not self.normalized_data:
            
            bin_col = self.normalized_data.copy() 
            
            bin_col = bin_col.where(bin_col <= 0, 1) 
            
            sum_data = bin_col.sum(axis=0)
            
            self.gene_calc = sum_data
    
    
    def gene_histograme(self, bins=100):
        
        """
        Plot a histogram of the number of genes detected per cell.

        Parameters
        ----------
        bins : int, default 100
            Number of histogram bins.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the histogram and an overlaid normal PDF.

        Notes
        -----
        - Requires `self.gene_calc` to be computed prior to calling.
        """

        fig, ax = plt.subplots(figsize=(8, 5))

        counts, bin_edges, patches = ax.hist(self.gene_calc, bins=bins, edgecolor="black", alpha=0.6)
        
        mu, sigma = np.mean(self.gene_calc), np.std(self.gene_calc)
        
        x = np.linspace(min(self.gene_calc), max(self.gene_calc), 1000)
        y = norm.pdf(x, mu, sigma)
        
        y_scaled = y * len(self.gene_calc) * (bin_edges[1] - bin_edges[0])
        
        ax.plot(x, y_scaled, "r-", linewidth=2, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
        
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of genes detected per cell")
        
        ax.set_xticks(np.linspace(min(self.gene_calc), max(self.gene_calc), 20))
        ax.tick_params(axis="x", rotation=90)
        
        ax.legend()
        
        return fig



    def gene_threshold(self, min_n:int | None, max_n:int | None):
        
        """
        Filter cells by gene-detection thresholds (min and/or max).

        Parameters
        ----------
        min_n : int or None
            Minimum number of detected genes required to keep a cell.
        max_n : int or None
            Maximum number of detected genes allowed to keep a cell.


        Update
        -------
            Filters `self.input_data`, `self.normalized_data`, `self.input_metadata`
            (and calls `average()` if `self.agg_normalized_data` exists).
          
        Side Effects
        ------------
           Raises ValueError if both bounds are None or if filtering removes all cells.
        """
        
        if min_n is not None and max_n is not None:
            mask = (self.gene_calc > min_n) & (self.gene_calc < max_n)
        elif min_n is None and max_n is not None:
            mask = self.gene_calc < max_n
        elif min_n is not None and max_n is None:
            mask = self.gene_calc > min_n
        else:
            raise ValueError("Lack of both min_n and max_n values")

            
        
        if self.input_data != None:
            
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing to reduce")
            
            self.input_data = self.input_data.loc[:, mask]

                
        
        if None is not self.normalized_data:
            
           
            
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing to reduce")
            
            self.normalized_data = self.normalized_data.loc[:, mask]
            

        if None is not self.input_metadata:
            
           
            if len([y for y in mask if y is False]) == 0:
                raise ValueError("Nothing to reduce")
            
            self.input_metadata = self.input_metadata.loc[mask, :].reset_index(drop = True)
            
            self.input_metadata = self.input_metadata.drop(columns=["drop"], errors="ignore")
            
        if None is not self.agg_normalized_data:
            self.average()


    def load_sparse_from_projects(self, normalized_data:bool = False):
        
        """
        Load sparse 10x-style datasets from stored project paths, concatenate them,
        and populate `input_data` / `normalized_data` and `input_metadata`.

        Parameters
        ----------
        normalized_data : bool, default False
            If True, store concatenated tables in `self.normalized_data`.
            If False, store them in `self.input_data` and normalization 
            is needed using normalize_data() method.


        Side Effects
        ------------
        - Reads each project using `load_sparse(...)` (expects matrix.mtx, genes.tsv, barcodes.tsv).
        - Concatenates all projects column-wise and sets `self.input_metadata`.
        - Replaces NaNs with zeros and updates `self.gene_calc`.
        """
        
        obj = self.objects
        
        full_data = pd.DataFrame()
        full_metadata = pd.DataFrame()
    
        for ke in obj.keys():
            print(ke)
            
            dt, met = load_sparse(path = obj[ke], name = ke)
            
            full_data = pd.concat([full_data, dt], axis = 1)
            full_metadata = pd.concat([full_metadata, met], axis = 0)
            
            
        full_data[np.isnan(full_data)] = 0
        
        if normalized_data == True:
            self.normalized_data = full_data
            self.input_metadata = full_metadata
        else:
            
            self.input_data = full_data
            self.input_metadata = full_metadata
            
            
        self.gene_calc()

    
    def rename_names(self, mapping: dict, slot:str = 'cell_names'):
        
        """
        Rename entries in `self.input_metadata[slot]` according to a provided mapping.

        Parameters
        ----------
        mapping : dict
            Dictionary with keys 'old_name' and 'new_name', each mapping to a list
            of equal length describing replacements.
        slot : str, default 'cell_names'
            Metadata column to operate on.

        Update
        -------
            Updates `self.input_metadata[slot]` in-place with renamed values.
              
        Raises
        ------
        ValueError
            If mapping keys are incorrect, lengths differ, or some 'old_name' values
            are not present in the metadata column.
        """
        
        if set(['old_name', 'new_name']) != set(mapping.keys()):
            raise ValueError(
                "Mapping dictionary must contain keys 'old_name' and 'new_name', "
                "each with a list of names to change."
            )    
            
        if len(mapping['old_name']) != len(mapping['new_name']):
            raise ValueError(
                "Mapping dictionary lists 'old_name' and 'new_name' "
                "must have the same length!"
            )
                   
            
        names_vector = list(self.input_metadata[slot])

        if not all(elem in names_vector for elem in list(mapping['old_name'])):
            raise ValueError(
                f"Some entries from 'old_name' do not exist in the names of slot {slot}."
            )
                
            
        replace_dict = dict(zip(mapping["old_name"], mapping["new_name"]))
        
        names_vector_ret = [replace_dict.get(item, item) for item in names_vector]
        
        self.input_metadata[slot] = names_vector_ret
        
        
        
    def rename_subclusters(self, mapping):
        
        """
        Rename labels stored in `self.subclusters_.subclusters` according to mapping.

        Parameters
        ----------
        mapping : dict
            Mapping with keys 'old_name' and 'new_name' (lists of equal length).

        
        Update
        -------
            Updates `self.subclusters_.subclusters` with renamed labels.

        Raises
        ------
        ValueError
            If mapping is invalid or old names are not present.
        """
        
        if set(['old_name', 'new_name']) != set(mapping.keys()):
            raise ValueError(
                "Mapping dictionary must contain keys 'old_name' and 'new_name', "
                "each with a list of names to change."
            )    
            
        if len(mapping['old_name']) != len(mapping['new_name']):
            raise ValueError(
                "Mapping dictionary lists 'old_name' and 'new_name' "
                "must have the same length!"
            )
                   
        names_vector = list(self.subclusters_.subclusters)

        if not all(elem in names_vector for elem in list(mapping['old_name'])):
            raise ValueError(
                "Some entries from 'old_name' do not exist in the subcluster names."
            )
                
            
        replace_dict = dict(zip(mapping["old_name"], mapping["new_name"]))
        
        names_vector_ret = [replace_dict.get(item, item) for item in names_vector]
        
        self.subclusters_.subclusters = names_vector_ret
        
        
    def save_sparse(self, 
                    path_to_save:str = os.getcwd(),
                    name_slot:str = 'cell_names', 
                    data_slot:str = 'normalized'):
        

        """
        Export data as 10x-compatible sparse files (matrix.mtx, barcodes.tsv, genes.tsv).

        Parameters
        ----------
        path_to_save : str, default current working directory
            Directory where files will be written.
        name_slot : str, default 'cell_names'
            Metadata column providing cell names for barcodes.tsv.
        data_slot : str, default 'normalized'
            Either 'normalized' (uses self.normalized_data) or 'count' (uses self.input_data).


        Raises
        ------
        ValueError
            If `data_slot` is not 'normalized' or 'count'.
        """
        
        names = self.input_metadata[name_slot]

        if data_slot.lower() == 'normalized':
            
            features = list(self.normalized_data.index)
            mtx = sparse.csr_matrix(self.normalized_data)
            
        
        elif data_slot.lower() == 'count':
            
            features = list(self.input_data.index)
            mtx = sparse.csr_matrix(self.input_data)
            
        else:
            raise ValueError("'data_slot' must be included in 'normalized' or 'count'")
        
        
        os.makedirs(path_to_save, exist_ok=True)
        
        mmwrite(os.path.join(path_to_save, "matrix.mtx"), mtx)

        pd.Series(names).to_csv(os.path.join(path_to_save, "barcodes.tsv"), index=False, header=False, sep="\t")
        
        pd.Series(features).to_csv(os.path.join(path_to_save, "genes.tsv"), index=False, header=False, sep="\t")
        
        
    
    def normalize_counts(self, normalize_factor: int = 100000, log_transform: bool = True):
        """
        Normalize raw counts to counts-per-(normalize_factor) 
        (e.g., CPM, TPM - depending on normalize_factor).
    
        Parameters
        ----------
        normalize_factor : int, default 100000
            Scaling factor used after dividing by column sums.
        log_transform : bool, default True
            If True, apply log2(x+1) transformation to normalized values.
    
        Update
        -------
            Sets `self.normalized_data` to normalized values (fills NaNs with 0).
    
        Raises
        ------
        ValueError
            If `self.input_data` is missing (cannot normalize).

        """
        if self.input_data is None:
            raise ValueError("Input data is missing, cannot normalize.")
    
        sum_col = self.input_data.sum()
        self.normalized_data = self.input_data.div(sum_col).fillna(0) * normalize_factor
    
        if log_transform:
            # log2(x + 1) to avoid -inf for zeros
            self.normalized_data = np.log2(self.normalized_data + 1)

      
    def statistic(self, cells=None, sets=None, min_exp:float = 0.01, min_pct:float = 0.1, n_proc:int=10):
        
        """
        Compute per-feature statistics (Mann–Whitney U) comparing target vs rest.

        This is a wrapper similar to `calc_DEG` tailored to use `self.normalized_data`
        and `self.input_metadata`. It returns per-feature statistics including p-values,
        adjusted p-values, means, variances, effect-size measures and fold-changes.

        Parameters
        ----------
        cells : list, 'All', dict, or None
            Defines the target cells or groups for comparison (several modes supported).
        sets : 'All', dict, or None
            Alternative grouping mode (operate on `self.input_metadata['sets']`).
        min_exp : float, default 0.01
            Minimum expression threshold used when filtering features.
        min_pct : float, default 0.1
            Minimum proportion of expressing cells in the target group required to test a feature.
        n_proc : int, default 10
            Number of parallel jobs to use.

        Returns
        -------
        pandas.DataFrame or dict
            Results DataFrame (or dict containing valid/control cells + DataFrame),
            similar to `calc_DEG` interface.

        Raises
        ------
        ValueError
            If neither `cells` nor `sets` is provided, or input metadata mismatch occurs.

        Notes
        -----
        - Multiple modes supported: single-list entities, 'All', pairwise dicts, etc.
        """

        def stat_calc(choose, feature_name):
            target_values = choose.loc[choose['DEG'] == 'target', feature_name]
            rest_values = choose.loc[choose['DEG'] == 'rest', feature_name]
 
            pct_valid = (target_values > 0).sum() / len(target_values)
            pct_rest = (rest_values > 0).sum() / len(rest_values)
 
            avg_valid = np.mean(target_values)
            avg_ctrl = np.mean(rest_values)
            sd_valid = np.std(target_values, ddof=1)
            sd_ctrl =  np.std(rest_values, ddof=1)
            esm = (avg_valid - avg_ctrl) /  np.sqrt(((sd_valid**2 + sd_ctrl**2) / 2))
 
            if np.sum(target_values) == np.sum(rest_values):
                p_val = 1.0
            else:
                _, p_val = stats.mannwhitneyu(target_values, rest_values, alternative='two-sided')
 
            return {
                'feature': feature_name,
                'p_val': p_val,
                'pct_valid': pct_valid,
                'pct_ctrl': pct_rest,
                'avg_valid': avg_valid,
                'avg_ctrl': avg_ctrl,
                'sd_valid': sd_valid,
                'sd_ctrl': sd_ctrl,
                'esm':esm
            }



        def prepare_and_run_stat(choose, valid_group, min_exp, min_pct, n_proc, factors):  
            
            
            
            tmp_dat = choose[choose['DEG'] == 'target']
            tmp_dat = tmp_dat.drop('DEG', axis = 1)
            
            counts = (tmp_dat > min_exp).sum(axis=0)
    
            total_count = tmp_dat.shape[0]
            
            
            info = pd.DataFrame({'featrue': list(tmp_dat.columns), 'pct':list(counts / total_count)})
            
            del tmp_dat
    
            drop_col = info['featrue'][info['pct'] <= min_pct]
            
            if len(drop_col) + 1 == len(choose.columns):
                drop_col = info['featrue'][info['pct'] == 0]
                
            del info
    
            choose = choose.drop(list(drop_col), axis = 1)
    
            results = Parallel(n_jobs=n_proc)(
                delayed(stat_calc)(choose, feature)
                for feature in tqdm(choose.columns[choose.columns != 'DEG'])
            )
    
            df = pd.DataFrame(results)
            df['valid_group'] = valid_group
            df.sort_values(by='p_val', inplace=True)
    
            num_tests = len(df)
            df['adj_pval'] = np.minimum(1, (df['p_val'] * num_tests) / np.arange(1, num_tests + 1))
    
            factors_df = factors.to_frame(name='factor')

            df = df.merge(factors_df, left_on='feature', right_index=True, how='left')

            df['FC'] = (df['avg_valid'] + df['factor']) / (df['avg_ctrl'] + df['factor'])
            df['log(FC)'] = np.log2(df['FC'])
            df['norm_diff'] = df['avg_valid'] - df['avg_ctrl']
            df = df.drop(columns=["factor"])
    
            return df


        choose = self.normalized_data.copy().T
        factors = self.normalized_data.copy().replace(0, np.nan).min(axis=1)
        factors[factors == 0] = min(factors[factors != 0])
        
        final_results = []


        if cells and isinstance(cells, list) and sets is None:
            print('\nAnalysis started...\nComparing selected cells to the whole set...')
            choose.index = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']

            labels = ['target' if idx in cells else 'rest' for idx in choose.index]
            valid = list(set(choose.index[[i for i, x in enumerate(labels) if x == 'target']]))
            
            choose['DEG'] = labels
            choose = choose[choose['DEG'] != 'drop']

            result_df = prepare_and_run_stat(choose.reset_index(drop=True), valid_group=valid, min_exp=min_exp, min_pct = min_pct, n_proc = n_proc, factors = factors)
            return {'valid_cells': valid, 'control_cells': 'rest', 'DEG': result_df}

        elif cells == 'All' and sets is None:
            print('\nAnalysis started...\nComparing each type of cell to others...')
            choose.index = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']
            unique_labels = set(choose.index)

            for label in tqdm(unique_labels):
                print(f'\nCalculating statistics for {label}')
                labels = ['target' if idx == label else 'rest' for idx in choose.index]
                choose['DEG'] = labels
                choose = choose[choose['DEG'] != 'drop']
                result_df = prepare_and_run_stat(choose.copy(), valid_group=label, min_exp=min_exp, min_pct = min_pct, n_proc = n_proc, factors = factors)
                final_results.append(result_df)

            return pd.concat(final_results, ignore_index=True)

        elif cells is None and sets == 'All':
            print('\nAnalysis started...\nComparing each set/group to others...')
            choose.index = self.input_metadata['sets']
            unique_sets = set(choose.index)

            for label in tqdm(unique_sets):
                print(f'\nCalculating statistics for {label}')
                labels = ['target' if idx == label else 'rest' for idx in choose.index]
                
                choose['DEG'] = labels
                choose = choose[choose['DEG'] != 'drop']
                result_df = prepare_and_run_stat(choose.copy(), valid_group=label, min_exp=min_exp, min_pct = min_pct, n_proc = n_proc, factors = factors)
                final_results.append(result_df)

            return pd.concat(final_results, ignore_index=True)

        elif cells is None and isinstance(sets, dict):
            print('\nAnalysis started...\nComparing groups...')
            choose.index = self.input_metadata['sets']

            group_list = list(sets.keys())
            if len(group_list) != 2:
                print('Only pairwise group comparison is supported.')
                return None


            labels = [
                'target' if idx in sets[group_list[0]] else
                'rest' if idx in sets[group_list[1]] else
                'drop'
                for idx in choose.index
            ]
            choose['DEG'] = labels
            choose = choose[choose['DEG'] != 'drop']

            result_df = prepare_and_run_stat(choose.reset_index(drop=True), valid_group=group_list[0], min_exp=min_exp, min_pct = min_pct, n_proc = n_proc, factors = factors)
            return result_df
        
        elif isinstance(cells, dict) and sets is None:
            print('\nAnalysis started...\nComparing groups...')
            choose.index = self.input_metadata['cell_names'] + ' # ' + self.input_metadata['sets']

            group_list = list(sets.keys())
            if len(group_list) != 2:
                print('Only pairwise group comparison is supported.')
                return None


            labels = [
                'target' if idx in sets[group_list[0]] else
                'rest' if idx in sets[group_list[1]] else
                'drop'
                for idx in choose.index
            ]
            
            choose['DEG'] = labels
            choose = choose[choose['DEG'] != 'drop']

            result_df = prepare_and_run_stat(choose.reset_index(drop=True), valid_group=group_list[0], min_exp=min_exp, min_pct = min_pct, n_proc = n_proc, factors = factors)
            
            return result_df.reset_index(drop = True)
        

        else:
            raise ValueError("You must specify either 'cells' or 'sets' (or both). None were provided, which is not allowed for this analysis.")



    def calculate_difference_markers(self, 
                                     min_exp = 0, 
                                     min_pct = 0.25, 
                                     n_proc=10, 
                                     force:bool = False):
        
        
        """
        Compute differential markers (var_data) if not already present.

        Parameters
        ----------
        min_exp : float, default 0
            Minimum expression threshold passed to `statistic`.
        min_pct : float, default 0.25
            Minimum percent expressed in target group.
        n_proc : int, default 10
            Parallel jobs.
        force : bool, default False
            If True, recompute even if `self.var_data` is present.


        Update
        -------
            Sets `self.var_data` to the result of `self.statistic(...)`.
        
        Raise
        ------
            ValueError if already computed and `force` is False.
        """
        
        if None is self.var_data or force:
        
            self.var_data = self.statistic(cells = 'All', sets = None, min_exp = min_exp, min_pct = min_pct, n_proc=n_proc)
        
        else:
            raise ValueError(
                "self.calculate_difference_markers() has already been executed. "
                "The results are stored in self.var. "
                "If you want to recalculate with different statistics, please rerun the method with force=True."
            )  
            
            
    def clustering_features(self, 
                            features_list: list | None, 
                            name_slot:str = 'cell_names', 
                            p_val: float = 0.05,
                            top_n: int = 25,
                            adj_mean: bool = True,
                            beta: float = 0.2):
        
        """
        Prepare clustering input by selecting marker features and optionally smoothing cell values
        toward group means.

        Parameters
        ----------
        features_list : list or None
            If provided, use this list of features. If None, features are selected
            from `self.var_data` (adj_pval <= p_val, positive logFC) picking `top_n` per group.
        name_slot : str, default 'cell_names'
            Metadata column used for naming.
        p_val : float, default 0.05
            Adjusted p-value cutoff when selecting features automatically.
        top_n : int, default 25
            Number of top features per valid group to keep if `features_list` is None.
        adj_mean : bool, default True
            If True, adjust cell values toward group means using `beta`.
        beta : float, default 0.2
            Adjustment strength toward group mean.

        Update
        ------
            Sets `self.clustering_data` and `self.clustering_metadata` to the selected subset,
            ready for PCA/UMAP/clustering.
        """
        
        if features_list is None or len(features_list) == 0:
            
            if None is self.var_data:            
                raise ValueError("Lack of 'self.var_data'. Use self.calculate_difference_markers() method first.")
            
            
            df_tmp = self.var_data[self.var_data['adj_pval'] <= p_val]
            df_tmp = df_tmp[df_tmp['log(FC)'] > 0]
            df_tmp = df_tmp.sort_values(['valid_group', 'esm'], ascending=[True, False]).groupby('valid_group').head(top_n)    

            feaures_list = list(set(df_tmp['feature']))
        
        data = self.get_partial_data(names = None, features = feaures_list, name_slot = name_slot)
        data_avg =  average(data)
        
        if adj_mean:
            data = adjust_cells_to_group_mean(data=data, data_avg=data_avg, beta=beta)
        
        self.clustering_data = data
          
        self.clustering_metadata = self.input_metadata
            
        
    def average(self):
        
        """
        Aggregate normalized data by (cell_name, set) pairs computing the mean per group.

        The method constructs new column names as "cell_name # set", averages columns
        sharing identical labels, and populates `self.agg_normalized_data` and `self.agg_metadata`.

        Update
        ------
            Sets `self.agg_normalized_data` (features x aggregated samples) and
            `self.agg_metadata` (DataFrame with 'cell_names' and 'sets').
        """
        
        wide_data = self.normalized_data
        
        wide_metadata = self.input_metadata
        
        new_names = wide_metadata['cell_names'] + ' # ' + wide_metadata['sets']
        
        wide_data.columns = list(new_names)
        
        aggregated_df = wide_data.T.groupby(wide_data.columns, axis=0).mean().T
    
        
        sets = [re.sub('.*# ', '' , x) for x in aggregated_df.columns]
        names = [re.sub(' #.*', '' , x) for x in aggregated_df.columns]
    
        aggregated_df.columns = names
        aggregated_metadata = pd.DataFrame({'cell_names':names, 'sets':sets})
        
        self.agg_metadata = aggregated_metadata
        self.agg_normalized_data = aggregated_df
        
        
    def estimating_similarity(self, 
                              method = 'pearson',
                              p_val: float = 0.05,
                              top_n: int = 25):
        
        """
        Estimate pairwise similarity and Euclidean distance between aggregated samples.

        Parameters
        ----------
        method : str, default 'pearson'
            Correlation method to use (passed to pandas.DataFrame.corr()).
        p_val : float, default 0.05
            Adjusted p-value cutoff used to select marker features from `self.var_data`.
        top_n : int, default 25
            Number of top features per valid group to include.


        Update
        -------
            Computes a combined table with per-pair correlation and euclidean distance
            and stores it in `self.similarity`.
        """
        
        if None is self.var_data:            
            raise ValueError("Lack of 'self.var_data'. Use self.calculate_difference_markers() method first.")
            
        if None is self.agg_normalized_data:
            self.average()      
            
            
        metadata = self.agg_metadata
        data = self.agg_normalized_data
        
        
        df_tmp = self.var_data[self.var_data['adj_pval'] <= p_val]
        df_tmp = df_tmp[df_tmp['log(FC)'] > 0]
        df_tmp = df_tmp.sort_values(['valid_group', 'esm'], ascending=[True, False]).groupby('valid_group').head(top_n)    

        
        data = data.loc[list(set(df_tmp['feature']))]
        
            
        if len(set(metadata['sets'])) > 1:
            data.columns = data.columns + ' # ' + [x for x in metadata['sets']]
        else:
            data = data.copy()
            
        
        scaler = StandardScaler()
        
        scaled_data = scaler.fit_transform(data)
        
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
            
        cor = scaled_df.corr(method=method)
        cor_df = cor.stack().reset_index()
        cor_df.columns = ['cell1', 'cell2', 'correlation']
        
        distances = pdist(scaled_df.T, metric='euclidean')
        dist_mat = pd.DataFrame(squareform(distances), index=scaled_df.columns, columns=scaled_df.columns)
        dist_df = dist_mat.stack().reset_index()
        dist_df.columns = ['cell1', 'cell2', 'euclidean_dist']
        
        full = pd.merge(cor_df, dist_df, on=['cell1', 'cell2'])
        
        full = full[full['cell1'] != full['cell2']]
        full = full.reset_index(drop = True)
        
        self.similarity = full



    def similarity_plot(self, 
                        split_sets = True, 
                        set_info:bool = True,
                        cmap='seismic', 
                        width = 12, 
                        height = 10):
        
        """
        Visualize pairwise similarity as a scatter plot.

        Parameters
        ----------
        split_sets : bool, default True
            If True and set information is present, split plotting area roughly into two halves to visualize cross-set pairs.
        set_info : bool, default True
            If True, keep the ' # set' annotation in labels; otherwise strip it.
        cmap : str, default 'seismic'
            Color map for correlation (hue).
        width : int, default 12
            Figure width.
        height : int, default 10
            Figure height.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If `self.similarity` is None.

        Notes
        -----
        - The function filters pairs by z-scored euclidean distance > 0 to focus on closer pairs.
        """
        
        if None is self.similarity:
            raise ValueError('Similarity data is missing. Please calculate similarity using self.estimating_similarity.')
        
        
        similarity_data = self.similarity
    
    
        if ' # ' in similarity_data['cell1'][0]:
            similarity_data['set1'] = [re.sub('.*# ', '' , x) for x in similarity_data['cell1']]
            similarity_data['set2'] = [re.sub('.*# ', '' , x) for x in similarity_data['cell2']]
            
            
        if split_sets == True and ' # ' in similarity_data['cell1'][0]:
            sets = list(set(list(similarity_data['set1'] ) + list(similarity_data['set2'])))
            
            mm = math.ceil(len(sets)/2)
            
            x_s = sets[0:mm]
            y_s = sets[mm:len(sets)]
            
            similarity_data = similarity_data[similarity_data['set1'].isin(x_s)]
            similarity_data = similarity_data[similarity_data['set2'].isin(y_s)]
            
            similarity_data = similarity_data.sort_values(['set1','set2'])

        if set_info == False and ' # ' in similarity_data['cell1'][0]:
            similarity_data['cell1'] = [re.sub(' #.*', '' , x) for x in similarity_data['cell1']]
            similarity_data['cell2'] = [re.sub(' #.*', '' , x) for x in similarity_data['cell2']]
            
            
        
        similarity_data['euclidean_zscore'] = -zscore(similarity_data['euclidean_dist']) 
    
        similarity_data = similarity_data[similarity_data['euclidean_zscore'] > 0]
    
        fig = plt.figure(figsize=(width, height))
        sns.scatterplot(
            data=similarity_data,
            x='cell1',
            y='cell2',
            hue='correlation',
            size='euclidean_zscore',
            sizes=(1, 100),
            palette=cmap,
            alpha=1,
            edgecolor='black'  
        )
    
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.xlabel('Cell 1')
        plt.ylabel('Cell 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.grid(True, alpha=0.6)
    
        plt.tight_layout()
            
        return fig 
    
    
    def spatial_similarity(self, 
                           set_info: bool = True,
                           bandwidth = 1,
                           n_neighbors = 5,  
                           min_dist = 0.1, 
                           legend_split = 2, 
                           point_size = 20, 
                           spread=1.0,              
                           set_op_mix_ratio=1.0,    
                           local_connectivity=1,    
                           repulsion_strength=1.0,  
                           negative_sample_rate=5,
                           threshold = 0.1,
                           width = 12, 
                           height = 10):
        
        """
        Create a spatial UMAP-like visualization of similarity relationships between samples.

        Parameters
        ----------
        set_info : bool, default True
            If True, retain set information in labels.
        bandwidth : float, default 1
            Bandwidth used by MeanShift for clustering polygons.
        n_neighbors, min_dist, spread, set_op_mix_ratio, local_connectivity, repulsion_strength, negative_sample_rate : parameters passed to UMAP.
        threshold : float, default 0.1
            Minimum text distance for label adjustment to avoid overlap.
        width : int, default 12
            Figure width.
        height : int, default 10
            Figure height.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If `self.similarity` is None.

        Notes
        -----
        - Builds a precomputed distance matrix combining correlation and euclidean distance,
          runs UMAP with metric='precomputed', then overlays cluster hulls (MeanShift + convex hull)
          and arrows to indicate nearest neighbors (minimal combined distance).
        """
        
        if None is self.similarity:
            raise ValueError('Similarity data is missing. Please calculate similarity using self.estimating_similarity.')
        
        similarity_data = self.similarity
     
        
        similarity_data['combo_dist'] = (1 - similarity_data['correlation']) *  similarity_data['euclidean_dist']
     
        # for nn target
        arrow_df = similarity_data.copy()
        arrow_df = similarity_data.loc[similarity_data.groupby('cell1')['combo_dist'].idxmin()]
            
            
        cells = sorted(set(similarity_data['cell1']) | set(similarity_data['cell2']))
        combo_matrix = pd.DataFrame(0, index=cells, columns=cells, dtype=float)
     
        for _, row in similarity_data.iterrows():
            combo_matrix.loc[row['cell1'], row['cell2']] = row['combo_dist']
            combo_matrix.loc[row['cell2'], row['cell1']] = row['combo_dist']  
        
        
     
            
        umap_model = umap.UMAP(
             n_components=2,          
             metric='precomputed',   
             n_neighbors=n_neighbors,          
             min_dist=min_dist,            
             spread=spread,              
             set_op_mix_ratio=set_op_mix_ratio,    
             local_connectivity=set_op_mix_ratio,    
             repulsion_strength=repulsion_strength,  
             negative_sample_rate=negative_sample_rate,  
             transform_seed=42,       
             init='spectral',         
             random_state=42,        
             verbose=True
             
         )
        
        coords = umap_model.fit_transform(combo_matrix.values)
        cell_names = list(combo_matrix.index)
        num_cells = len(cell_names)
        palette = sns.color_palette("tab20c", num_cells)

        if '#' in cell_names[0]:
            avsets = set([re.sub('.*# ', '', x) for x in similarity_data['cell1']] +
                         [re.sub('.*# ', '', x) for x in similarity_data['cell2']])
            num_sets = len(avsets)
            color_indices = [i * len(palette) // num_sets for i in range(num_sets)]
            color_mapping_sets = {set_name: palette[i] for i, set_name in zip(color_indices, avsets)}
            color_mapping = {name: color_mapping_sets[re.sub('.*# ', '', name)] for i, name in enumerate(cell_names)}
        else:
            color_mapping = {name: palette[i] for i, name in enumerate(cell_names)}
            
            
      
        meanshift = MeanShift(bandwidth=bandwidth)  
        labels = meanshift.fit_predict(coords)

        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()

        unique_labels = set(labels)
        cluster_palette = sns.color_palette("hls", len(unique_labels))

        for label in unique_labels:
            if label == -1:
                continue
            cluster_coords = coords[labels == label]
            if len(cluster_coords) < 3:
                continue

            hull = ConvexHull(cluster_coords)
            hull_points = cluster_coords[hull.vertices]

            centroid = np.mean(hull_points, axis=0)
            expanded = hull_points + 0.05 * (hull_points - centroid)

            poly = Polygon(
                expanded,
                closed=True,
                facecolor=cluster_palette[label],
                edgecolor='none',
                alpha=0.2,
                zorder=1
            )
            ax.add_patch(poly)

        texts = []
        for i, (x, y) in enumerate(coords):
            plt.scatter(x, y, s=200, color=color_mapping[cell_names[i]], edgecolors='black', linewidths=0.5, zorder=2)
            texts.append(ax.text(x, y, str(i), ha='center', va='center',
                                 fontsize=8, color='black'))
            

        
        def dist(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        texts_to_adjust = []
        for i, t1 in enumerate(texts):
            for j, t2 in enumerate(texts):
                if i >= j:
                    continue
                d = dist((t1.get_position()[0], t1.get_position()[1]),
                         (t2.get_position()[0], t2.get_position()[1]))
                if d < threshold:
                    if t1 not in texts_to_adjust:
                        texts_to_adjust.append(t1)
                    if t2 not in texts_to_adjust:
                        texts_to_adjust.append(t2)

        adjust_text(
            texts_to_adjust,
            expand_text=(1.0, 1.0),
            force_text=0.9,
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.1),
            ax=ax
        )  
              
            
        for _, row in arrow_df.iterrows():
            try:
                idx1 = cell_names.index(row['cell1'])
                idx2 = cell_names.index(row['cell2'])
            except ValueError:
                continue
            x1, y1 = coords[idx1]
            x2, y2 = coords[idx2]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color='gray',
                                    linewidth=1.5, alpha=0.5, mutation_scale=12, zorder=0)
            ax.add_patch(arrow)
        
        
        if set_info == False and ' # ' in cell_names[0]:
            
            
            legend_elements = [
                Patch(facecolor=color_mapping[name], edgecolor='black', label=f"{i} – {re.sub(' #.*', '', name)}")
                for i, name in enumerate(cell_names)
            ]
            
        
        else:
            
            legend_elements = [
                Patch(facecolor=color_mapping[name], edgecolor='black', label=f"{i} – {name}")
                for i, name in enumerate(cell_names)
            ]
        
        plt.legend(handles=legend_elements, title="Cells", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=legend_split)

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(False)
        plt.show()
     
             
        return fig
    
    
    # subclusters part
    
    def subcluster_prepare(self, features: list, cluster: str):

        """
        Prepare a `Clustering` object for subcluster analysis on a selected parent cluster.

        Parameters
        ----------
        features : list
            Features to include for subcluster analysis.
        cluster : str
            Parent cluster name (used to select matching cells).

        Update
        ------
            Initializes `self.subclusters_` as a new `Clustering` instance containing the
            reduced data for the given cluster and stores `current_features` and `current_cluster`.
        """
        
        dat = self.normalized_data
        dat.columns = list(self.input_metadata['cell_names'])
        
        dat = reduce_data(self.normalized_data,
                    features = features,
                    names = [cluster])


        self.subclusters_ = Clustering(data = dat, metadata = None)
        
        self.subclusters_.current_features = features
        self.subclusters_.current_cluster = cluster

        


    def define_subclusters(self, 
                          umap_num:int = 2,
                          eps:float = 0.5, 
                          min_samples:int = 10,
                          bandwidth:int = 1,
                          n_neighbors:int = 5,  
                          min_dist:float = 0.1, 
                          legend_split:int = 2, 
                          spread:float=1.0,              
                          set_op_mix_ratio:float=1.0,    
                          local_connectivity:int=1,    
                          repulsion_strength:float=1.0,  
                          negative_sample_rate:int=5,  
                          width = 8, 
                          height = 6):
        
        """
        Compute UMAP and DBSCAN clustering within a previously prepared subcluster dataset.

        Parameters
        ----------
        umap_num : int, default 2
            Number of UMAP dimensions to compute.
        eps : float, default 0.5
            DBSCAN eps parameter.
        min_samples : int, default 10
            DBSCAN min_samples parameter.
        bandwidth, n_neighbors, min_dist, legend_split, spread, set_op_mix_ratio, local_connectivity, repulsion_strength, negative_sample_rate, width, height :
            Additional parameters passed to UMAP / plotting / MeanShift as appropriate.

      
        Update
        -------
            Stores cluster labels in `self.subclusters_.subclusters`.
        
        Raises
        ------
        RuntimeError
            If `self.subclusters_` has not been prepared.
        """
        
        if self.subclusters_ is None:
            raise RuntimeError("Nothing to return. 'self.subcluster_prepare' was not conducted!")
                
        self.subclusters_.perform_UMAP(factorize = False,
                                                umap_num = umap_num,
                                                pc_num = 0,
                                                harmonized = False,
                                                bandwidth = bandwidth,
                                                n_neighbors = n_neighbors,  
                                                min_dist = min_dist, 
                                                legend_split = legend_split, 
                                                spread = spread,              
                                                set_op_mix_ratio = set_op_mix_ratio,    
                                                local_connectivity = local_connectivity,    
                                                repulsion_strength = repulsion_strength,  
                                                negative_sample_rate = negative_sample_rate,  
                                                width = width, 
                                                height = height)

        self.subclusters_.find_clusters_UMAP(umap_n = umap_num, 
                                    eps = eps, 
                                    min_samples = min_samples,
                                    width = width,
                                    height = height)

        clusters = self.subclusters_.return_clusters(clusters = 'umap')

        self.subclusters_.subclusters = [str(x) for x in list(clusters)]



    def subcluster_features_scatter(self,
                                    colors = 'viridis', 
                                    hclust = 'complete', 
                                    img_width = 3, 
                                    img_high = 5, 
                                    label_size = 6, 
                                    size_scale = 70,
                                    x_lab = 'Genes', 
                                    legend_lab = 'normalized'):
        
        """
        Create a features-scatter visualization for the subclusters (averaged and occurrence).

        Parameters
        ----------
        colors : str, default 'viridis'
            Colormap name passed to `features_scatter`.
        hclust : str or None
            Hierarchical clustering linkage to order rows/columns.
        img_width, img_high : float
            Figure size.
        label_size : int
            Font size for labels.
        size_scale : int
            Bubble size scaling.
        x_lab : str
            X axis label.
        legend_lab : str
            Colorbar label.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If subcluster preparation/definition has not been run.

        """
        
        if self.subclusters_ is None or self.subclusters_.subclusters is None:
            raise RuntimeError("Nothing to return. 'self.subcluster_prepare' -> 'self.define_subclusters' pip was not conducted!")
                

        dat = self.normalized_data
        dat.columns = list(self.input_metadata['cell_names'])
        
        dat = reduce_data(self.normalized_data,
                    features = self.subclusters_.current_features ,
                    names = [self.subclusters_.current_cluster])

        
        dat.columns = self.subclusters_.subclusters
        
      
        avg = average(dat)
        occ = occurrence(dat)
        
        
        
        scatter = features_scatter(expression_data = avg, 
                                   occurence_data=occ,
                                     features = None, 
                                     metadata_list = None, 
                                     colors = colors, 
                                     hclust = hclust, 
                                     img_width = img_width, 
                                     img_high = img_high, 
                                     label_size = label_size, 
                                     size_scale = size_scale,
                                     x_lab = x_lab, 
                                     legend_lab = legend_lab)
        
        return scatter


    def subcluster_DEG_scatter(self,
                                    top_n = 3,
                                    min_exp = 0, 
                                    min_pct = 0.25, 
                                    p_val = 0.05,
                                    colors = 'viridis', 
                                    hclust = 'complete', 
                                    img_width = 3, 
                                    img_high = 5, 
                                    label_size = 6, 
                                    size_scale = 70,
                                    x_lab = 'Genes', 
                                    legend_lab = 'normalized',
                                    n_proc=10):
        
        """
        Plot top differential features (DEGs) for subclusters as a features-scatter.

        Parameters
        ----------
        top_n : int, default 3
            Number of top features per subcluster to show.
        min_exp, min_pct, p_val : float
            Filtering thresholds used when computing DEGs.
        colors, hclust, img_width, img_high, label_size, size_scale, x_lab, legend_lab :
            Passed to `features_scatter`.
        n_proc : int, default 10
            Parallel jobs used for DEG calculation.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If subcluster preparation/definition has not been run.

        Notes
        -----
        - Internally calls `calc_DEG` (or equivalent) to obtain statistics, filters
          by p-value and effect-size, selects top features per valid group and plots them.
        """
        
        if self.subclusters_ is None or self.subclusters_.subclusters is None:
            raise RuntimeError("Nothing to return. 'self.subcluster_prepare' -> 'self.define_subclusters' pip was not conducted!")
                
        dat = self.normalized_data
        dat.columns = list(self.input_metadata['cell_names'])
        
        dat = reduce_data(self.normalized_data,
                    names = [self.subclusters_.current_cluster])

        
        dat.columns = self.subclusters_.subclusters
        

        stats = calc_DEG(dat, 
                         metadata_list = None, 
                         entities='All', 
                         sets=None, 
                         min_exp = min_exp, 
                         min_pct = min_pct, 
                         n_proc=n_proc)


        stats = stats[stats['p_val'] <= p_val]
        
        stats = stats.sort_values(['esm', 'log(FC)'], ascending=[False, False]).groupby('valid_group').head(top_n)
                          
                       
        dat = reduce_data(dat,
                          features = list(set(stats['feature'])))          
                       
        avg = average(dat)
        occ = occurrence(dat)


        scatter = features_scatter(expression_data = avg, 
                                   occurence_data=occ,
                                     features = None, 
                                     metadata_list = None, 
                                     colors = colors, 
                                     hclust = hclust, 
                                     img_width = img_width, 
                                     img_high = img_high, 
                                     label_size = label_size, 
                                     size_scale = size_scale,
                                     x_lab = x_lab, 
                                     legend_lab = legend_lab)

        return scatter


    def accept_subclusters(self):
        
        """
        Commit subcluster labels into the main `input_metadata` by renaming cell names.

        The method replaces occurrences of the parent cluster name in `self.input_metadata['cell_names']`
        with the expanded names that include subcluster suffixes (via `add_subnames`),
        then clears `self.subclusters_`.

        
        Update
        ------
            Modifies `self.input_metadata['cell_names']`.
            Resets `self.subclusters_` to None.
            
        Raises
        ------
        RuntimeError
            If `self.subclusters_` is not defined or subclusters were not computed.
        """
        
        if self.subclusters_ is None or self.subclusters_.subclusters is None:
            raise RuntimeError("Nothing to return. 'self.subcluster_prepare' -> 'self.define_subclusters' pip was not conducted!")
                 
        
        new_meta = add_subnames(list(self.input_metadata['cell_names']), 
                                parent_name = self.subclusters_.current_cluster, 
                                new_clusters = self.subclusters_.subclusters)


        self.input_metadata['cell_names'] = new_meta
        
        self.subclusters_ = None
     
 
    


        

