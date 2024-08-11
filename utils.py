from used_packages import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tensor(data, dtype=torch.float32, device=device):
    return torch.tensor(data, dtype=dtype, device=device)

#%%

def eval_nn(test_data, model):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = 100 * correct / total

    return accuracy, cm, all_preds

#%%

def conf(cm, display_cm=False):

    ind = range(1, 16)
    c = pd.DataFrame(cm, index=ind, columns=ind)
    sum = np.sum(c, axis=1).values
    acc_individual = pd.DataFrame(np.round((np.diag(c) / sum)*100, 2), index=ind)
    accuracy = np.diag(c).sum() / cm.sum()
    print(f'\n=============================')
    print(f'Overall Accuracy {accuracy*100 :.2f}%')
    print(f'Per-class Accuracy (%)\n============================= {acc_individual}\n')

    if display_cm is not False:
        # Display the confusion matrix
        fig, ax = plt.subplots(figsize=(7, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.show()
    return acc_individual


#%%

def load_nn(n_images, model_path):
    from models import SimpleNN
    # Recreate the model architecture
    model = SimpleNN(n_images)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

#%%

def plot_image(image, title, fig_size=(10, 10), cmap='gray', fs=25, vmin=None, vmax=None, save=False):
    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=fs)
    plt.xticks([]), plt.yticks([])
    if save is True:
        plt.savefig(f'{title}.png', bbox_inches='tight')
    plt.show()

#%%

def map_roi(image, color_map, tr_ind, ts_ind, roi_path):
    import pandas as pd

    # Training DataFrame with X and Y coordinates of pixels to be colored
    image_t = np.zeros_like(image.astype(float))

    image_scaled = ((image - image.min())/(image.max() - image.min()))*2

    # Convert the grayscale image to RGB
    rgb_image_full = np.stack((image_scaled,)*3, axis=-1)
    rgb_image_tr = np.stack((image_t,)*3, axis=-1)
    labels_tr = np.stack(np.zeros_like(image_t.T), axis=-1)

    # Set the specified pixels to colors from avaris
    for label, i in enumerate(range(len(tr_ind))):
        df = pd.DataFrame(tr_ind[i])
        for index, row in df.iterrows():
            rgb_image_tr[row['Y']-1, row['X']-1] = color_map[i].tolist()  # put colored labels on segmented train image
            rgb_image_full[row['Y']-1, row['X']-1] = color_map[i].tolist() # color the lidar image
            labels_tr[row['Y']-1, row['X']-1] = label+1 # create train label image

    # Testing DataFrame with X and Y coordinates of pixels to be colored
    image_ts = np.zeros_like(image.astype(float)).astype(float)

    rgb_image_ts = np.stack((image_ts,)*3, axis=-1)
    labels_ts = np.stack(np.zeros_like(image_ts.T), axis=-1)

    # Set the specified pixels to colors from avaris
    for label, i in enumerate(range(len(ts_ind))):
        df = pd.DataFrame(ts_ind[i])
        for index, row in df.iterrows():
            rgb_image_ts[row['Y']-1, row['X']-1] = color_map[i].tolist()  # put colored labels on segmented test image
            rgb_image_full[row['Y']-1, row['X']-1] = color_map[i].tolist() # color the lidar image
            labels_ts[row['Y']-1, row['X']-1] = label+1 # create test label image

    labels_tr = np.array(labels_tr)
    labels_ts = np.array(labels_ts)
    return labels_tr, labels_ts, rgb_image_full, rgb_image_tr, rgb_image_ts, rgb_image_full
#%%

# ensures reproducibility and deterministic behaviour for the NN model
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False