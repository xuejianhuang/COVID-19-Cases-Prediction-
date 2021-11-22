import matplotlib.pyplot  as plt
import torch
import  csv

def plot_learning_curve(train_loss,dev_loss, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, c='tab:red', label='train')
    plt.plot(dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()
def plot_pred(preds, targets):
    ''' Plot prediction of your DNN '''
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, 35], [-0.2, 35], c='b')
    plt.xlim(-0.2, 35)
    plt.ylim(-0.2, 35)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])