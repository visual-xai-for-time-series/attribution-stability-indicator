import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import umap.umap_ as umap_
except ImportError:
    import umap as umap_


def generate_histogram(data, labels=None, cmap='tab10'):
    for name, exp in data.items():
        score = np.sum([x[0] for x in exp])
        
        data_labels = np.array([x[2] for x in exp])
        gt = data_labels[:,0]
        p1 = data_labels[:,1]
        p2 = data_labels[:,2]
        
        acc1 = np.round(np.sum(gt == p1) / len(gt), 4)
        acc2 = np.round(np.sum(gt == p2) / len(gt), 4)
        
        flips = np.sum((p1 != p2).astype(int))
        
        print(f'Name: {name}')
        print(f'ASI: {score:.3f}, Flips: {flips}, Accuracy before {acc1}, after {acc2}')
        
        colors1 = gt
        colors2 = p1
        colors3 = p2
    
        d = [d[0] for d in exp]
        y_lim = len(d) / 5
        
        hist, bins = np.histogram(d, bins=50)
        bin_indices = np.digitize(d, bins)
        binned_indices = [np.where(bin_indices == i)[0] for i in range(1, len(bins))]
        
        def create_bar_for_colors(colors):
            unique_colors = np.unique(colors)
            stacked_data = np.zeros((len(binned_indices), len(unique_colors)))
            for i, x in enumerate(binned_indices):
                unique, counts = np.unique(colors[x], return_counts=True)
                stacked_data[i][unique] += counts

            bottom_line = np.zeros(len(stacked_data))
            stacked_data = np.transpose(stacked_data)
            for i, d in enumerate(stacked_data):
                plt.bar(list(range(len(d))), d, bottom=bottom_line, color=mpl.colormaps[cmap](i), alpha=0.7)
                bottom_line += d
        
        plt.figure()
        plt.suptitle(f'Name: {name} - ASI: {score:.3f}, Flips: {flips}')
        
        if labels == None:
            labels = list(np.unique(colors1))
        labels = [f'{x:>4}' for x in labels]
        handles = [plt.Rectangle((0,0),1,1, color=mpl.colormaps[cmap](i)) for i in range(len(labels))]
        
        plt.subplot(1, 3, 1)
        plt.ylim(0, y_lim)
        
        create_bar_for_colors(colors1)
        
        plt.legend(handles, labels)
        plt.title('Ground Truth')
            
        plt.subplot(1, 3, 2)
        plt.ylim(0, y_lim)
        
        create_bar_for_colors(colors2)
        
        plt.legend(handles, labels)
        plt.title('Initial Prediction')
        
        plt.subplot(1, 3, 3)
        plt.ylim(0, y_lim)
        
        create_bar_for_colors(colors3)
        
        plt.legend(handles, labels)
        plt.title('Perturbed Prediction')
        
        plt.show()
        

def generate_projections(data, labels=None, cmap='tab10'):
    for name, exp in data.items():
        score = np.sum([x[0] for x in exp])

        data_labels = np.array([x[2] for x in exp])
        gt = data_labels[:,0]
        p1 = data_labels[:,1]
        p2 = data_labels[:,2]
        
        acc1 = np.round(np.sum(gt == p1) / len(gt), 4)
        acc2 = np.round(np.sum(gt == p2) / len(gt), 4)
        
        flips = np.sum((p1 != p2).astype(int))
        
        print(f'Name: {name}')
        print(f'ASI: {score:.3f}, Flips: {flips}')
    
        data = np.array([x[1] for x in exp])

        colors1 = gt
        colors2 = p1
        colors3 = p2
        
        colors1_mapped = np.array([mpl.colormaps[cmap](x) for x in colors1])
        colors2_mapped = np.array([mpl.colormaps[cmap](x) for x in colors2])
        colors3_mapped = np.array([mpl.colormaps[cmap](x) for x in colors3])

        umap_reducer = umap_.UMAP(random_state=13)
        umap_data = umap_reducer.fit_transform(data)

        plt.figure()
        plt.suptitle(f'Name: {name} - ASI: {score:.3f}, Flips: {flips}')
        
        if labels == None:
            labels = list(np.unique(colors1))
        labels = [f'{x:>4}' for x in labels]
        handles = [plt.Rectangle((0,0),1,1, color=mpl.colormaps[cmap](i)) for i in range(len(labels))]
        
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(umap_data[:,0], umap_data[:,1], c=colors1_mapped, alpha=0.7)
        plt.legend(handles, labels)
        plt.title('Ground Truth')
        
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(umap_data[:,0], umap_data[:,1], c=colors2_mapped, alpha=0.7)
        plt.legend(handles, labels)
        plt.title('Initial Prediction')
        
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(umap_data[:,0], umap_data[:,1], c=colors3_mapped, alpha=0.7)
        plt.legend(handles, labels)
        plt.title('Perturbed Prediction')
        
        plt.show()
