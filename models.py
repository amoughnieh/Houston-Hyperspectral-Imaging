from used_packages import *
from utils import device, tensor, eval_nn

def svm_model(X_train, y_train, X_test, y_test, C, kernel='rbf', degree=3, bias=None, verbose=False):
    np.random.seed(0)
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    count = 0
    acc_max = []
    conf_max = []
    bias_max = []
    y_max = []
    for c in C:
        acc = []
        conf_matrix_all = []
        y = []
        for bia in bias:
            model_svm = SVC(C=c, kernel=kernel, degree=degree, coef0=bia).fit(X_train, y_train)
            y_p = model_svm.predict(X_test)
            a = accuracy_score(y_test, y_p)*100
            co = confusion_matrix(y_test, y_p)
            acc.append(a)
            conf_matrix_all.append(co)
            y.append(y_p)
            count += 1
            if verbose is not False:
                print(f'Model {count}/{len(bias)*len(C)} - C: {c :.4f}, bias: {bia}, accuracy: {a :.2f}%')
        indmx = np.argmax(acc)
        acc_max.append(acc[indmx])
        conf_max.append(conf_matrix_all[indmx])
        bias_max.append(bias[indmx])
        y_max.append(y[indmx])

    ind_max = np.argmax(acc_max)
    acc_opt = acc_max[ind_max]
    conf_matrix_optm = conf_max[ind_max]
    bias_optm = bias_max[ind_max]
    C_opt = C[ind_max]
    y_opt = y_max[ind_max]
    print(f'Best Model - accuracy: {acc_opt :.2f}%, C: {C_opt :.4f}, bias: {bias_optm}')
    return acc_opt, C_opt, bias_optm, conf_matrix_optm, y_opt

#%%
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 15)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = self.fc9(x)
        return x

#%%

def simple_nn(X_train, y_train, X_test, y_test, tol, steps=[0.001], batch_size=64, penalties=[1e-5], max_iter=900, verbose=False, disp_cm=False, opt='Adam', model_name='name', seed=42):
    from utils import set_seed

    set_seed(seed)

    from data import dataset_loader
    from utils import conf

    train_loader, test_loader = dataset_loader(X_train, y_train, X_test, y_test, batch_size)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    confs = []
    accuracies = []
    pens = []
    step_size = []
    y_preds = []
    total_time = []
    model_count = 1

    best_accuracy = 0.0  # Variable to track the best accuracy
    best_model_state = None  # Variable to store the state dict of the best model

    for step in steps:
        for penalty in penalties:
            start_time = time.time()
            print('\n==========================================')
            print(f'Begin Training:  Model {model_count}/{len(steps)*len(penalties)} ')
            print(f'Learning rate: {step} - Penalty {penalty:.2e}')
            print('==========================================')
            model_count += 1

            model = SimpleNN(X_train.shape[1]).to(device)
            criterion = nn.CrossEntropyLoss()
            if opt == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=step, weight_decay=penalty, momentum=0.9, nesterov=True)
            else:
                optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=penalty)

            running_loss = 1
            epoch = 0

            while (running_loss / len(train_loader) > tol) and epoch <= max_iter:

                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                epoch += 1
                if verbose == 1 or verbose == 2:
                    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

            accuracy, cm, all_preds = eval_nn(test_loader, model)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = f'saved models/best_model - {model_name} - acc {best_accuracy/100 :.4f} epoch {epoch-1} - lr {step} - regul {penalty}.pth'  # File path to save the best model
                best_model_state = model.state_dict()  # Save the best model state
                torch.save(best_model_state, best_model_path)  # Save the state dict to a file

            if running_loss / len(train_loader) <= tol:
                converged = 'Converged'
            else:
                converged = 'Not Converged'

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('*********************************************************************************')
            print(f'End Training: {converged} - Epochs {epoch-1}')
            print(f'Accuracy: {accuracy:.2f}% - Learning rate: {step} - Penalty {penalty:.2e} - Elapsed Time: {elapsed_time:.2f}s')
            print('*********************************************************************************')
            total_time.append(elapsed_time)

            if verbose == 2 or verbose == 3:
                # Display the confusion matrix
                conf(cm, display_cm=disp_cm)

            confs.append(cm)
            accuracies.append(accuracy)
            pens.append(penalty)
            step_size.append(step)
            y_preds.append(all_preds)

    ind_max_nn = np.argmax(accuracies)
    conf_max = confs[ind_max_nn]
    pen_max = pens[ind_max_nn]
    step_max = step_size[ind_max_nn]
    y_pred_max = y_preds[ind_max_nn]
    acc_max_nn = accuracies[ind_max_nn]

    dict_opt = {'conf_opt': conf_max,
                'pen_opt': pen_max,
                'step_opt': step_max,
                'y_pred_opt': y_pred_max,
                'acc_opt': acc_max_nn}

    print(f'Best model: Accuracy: {acc_max_nn:.2f}%, Penalty: {pen_max}, Step size: {step_max}')
    print(f'******* Total Time: {np.sum(total_time)/60:.2f} minutes *******')

    return dict_opt


#%%

