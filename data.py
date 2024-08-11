from used_packages import *


def import_data(path, hdr_path=None):
    if hdr_path is not None:
        image = spectral.envi.open(hdr_path, image=path)
        image = np.array(image.open_memmap(writeable=True))

    else:
        with rasterio.open(path) as src:
            image = src.read()
    return image


#%%
import re

def extract_roi_data(file_path):
    roi_data = []
    current_roi = None
    capture_points = False

    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    for line in file_content:
        line = line.strip()
        if line.startswith(';'):
            if 'ROI name' in line:
                if current_roi:
                    roi_data.append(current_roi)
                current_roi = {
                    'name': re.search(r'ROI name:\s*(.*)', line).group(1),
                    'rgb_value': None,
                    'npts': None,
                    'points': []
                }
                capture_points = False
            elif 'ROI rgb value' in line and current_roi:
                rgb_str = re.search(r'ROI rgb value:\s*\{(.*)\}', line).group(1)
                current_roi['rgb_value'] = list(map(int, rgb_str.split(',')))
            elif 'ROI npts' in line and current_roi:
                current_roi['npts'] = int(re.search(r'ROI npts:\s*(\d+)', line).group(1))
                capture_points = True
        elif capture_points and current_roi and line:
            match = re.match(r'(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)\s+([\d.-]+)', line) #capturing the train set
            if match:
                point = {
                    'ID': int(match.group(1)),
                    'X': int(match.group(2)),
                    'Y': int(match.group(3)),
                    'Lat': float(match.group(4)),
                    'Lon': float(match.group(5))
                }
                current_roi['points'].append(point)
            else: #capturing the test set
                match = re.match(r'(\d+)\s+(\d+)\s+(\d+)', line)
                point = {
                    'ID': int(match.group(1)),
                    'X': int(match.group(2)),
                    'Y': int(match.group(3)),
                }
                current_roi['points'].append(point)

    if current_roi:
        roi_data.append(current_roi)

    return roi_data

#%%

def train_test(X, tr_labels, ts_labels):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    train_indices = np.argwhere(tr_labels.ravel()).flatten()
    test_indices = np.argwhere(ts_labels.ravel()).flatten()
    try:
        X_tr = scaler.fit_transform(X.cpu().numpy().T[train_indices])
        y_tr = tr_labels.ravel()[train_indices]
        X_ts = scaler.fit_transform(X.cpu().numpy().T[test_indices])
        y_ts = ts_labels.ravel()[test_indices]
    except:
        X_tr = scaler.fit_transform(X.T[train_indices])
        y_tr = tr_labels.ravel()[train_indices]
        X_ts = scaler.fit_transform(X.T[test_indices])
        y_ts = ts_labels.ravel()[test_indices]
    return X_tr, y_tr, X_ts, y_ts

#%%

def dataset_loader(X_train, y_train, X_test, y_test, batch_size):
    from torch.utils.data import DataLoader, TensorDataset
    from utils import device

    X_train_sc = X_train
    X_test_sc = X_test

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_sc, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long).to(device)

    batch_size = batch_size

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader