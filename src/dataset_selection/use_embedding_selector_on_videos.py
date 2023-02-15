
#%%

import os
import glob
from embedding_selector import embedding_selector as es

#%%

video_dirs = [r'D:\PlacSeg\data\SourceVideos\LowCA_NoMg', r'D:\PlacSeg\data\SourceVideos\Microscopy']
target_dir = r'D:\PlacSeg\data\To_Be_Annotated_Images\yolov8_training_imgs'
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)

videos = []
for video_dir in video_dirs:
    videos += glob.glob(os.path.join(video_dir, '*.*'))
videos = [s for s in videos if '.csv' not in s]

#%%
imgs = es.use_embedding_select_on_videos(target_dir, videos, nimgs_embedding=1000,
                                        target_number_of_imgs=10, 
                                        n_clusters=10, 
                                        n_components_pca=200, n_components_umap=15, 
                                        n_neighbors=300, min_dist=0.05, metric='euclidean',
                                        save_imgs = True, resize=True, target_dim=(75,75), 
                                        save_target_dim=(640,640), plot=False)

# %%
