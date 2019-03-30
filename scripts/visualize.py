import matplotlib.pyplot as plt

SOURCE = '../log/train.log'
FOLD = 1

def extract_values(path):
    log_file = open(path, 'r')
    train = {}
    test = {}

    curr_train_loss = []
    curr_train_acc = []
    curr_test_loss = []
    curr_test_acc = []
    for line in log_file:
        items = line.strip().split()
        if len(items) == 17 and items[-2] == 'val_acc:':
            curr_train_loss.append(float(items[7]))
            curr_train_acc.append(float(items[10]))
            curr_test_loss.append(float(items[13]))
            curr_test_acc.append(float(items[16]))
        elif len(items) == 4 and items[0] == 'Cross':
            key = 'fold_' + items[-1]
            train[key + '_loss'] = curr_train_loss
            train[key + '_acc'] = curr_train_acc
            test[key + '_loss'] = curr_test_loss
            test[key + '_acc'] = curr_test_acc
            curr_train_loss = []
            curr_train_acc = []
            curr_test_loss = []
            curr_test_acc = []

    log_file.close()
    return train, test

def main():
    train, test = extract_values(SOURCE)
    key = 'fold_' + str(FOLD)
    train_loss = train[key + '_loss']
    train_acc = train[key + '_acc']
    test_loss = test[key + '_loss']
    test_acc = test[key + '_acc']
    epoch = [elem for elem in range(1, len(train_loss) + 1)]

    fig, axs = plt.subplots(1, 2, figsize = (10, 5), constrained_layout = True)
    fig.suptitle('Curve for ' + key)

    axs[0].plot(epoch, train_loss, c = 'blue', label = 'train_loss')
    axs[0].plot(epoch, test_loss, c = 'red', label = 'test_loss')
    axs[0].set_title('Loss Curve')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend()

    axs[1].plot(epoch, train_acc, c = 'blue', label = 'train_acc')
    axs[1].plot(epoch, test_acc, c = 'red', label = 'test_acc')
    axs[1].set_title('Accuracy Curve')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend()

    plt.show()

if __name__ == '__main__':
    main()
