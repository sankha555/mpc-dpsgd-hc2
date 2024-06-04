import matplotlib.pyplot as plt  
import numpy as np

def parse_results_vanilla(filepath, num_epochs):
    print(f"Parsing {filepath}...")

    epochs = [e for e in range(1, num_epochs+1)]

    test_losses = []
    train_losses = []

    test_accs = []
    train_accs = []

    lines = []
    with open(filepath, "r") as file:
        lines = file.readlines() 

    final_lines = []
    for line in lines:
        if 'Train' in line or 'Test' in line:
            final_lines.append(line)
        
    assert len(final_lines) == 2*num_epochs 

    for e in range(num_epochs):
        train_line = final_lines[2*e]
        loss_part = train_line.split('Loss: ')[1]
        loss, acc_part = loss_part.split('Acc@1:')
        acc, eps_part = acc_part.split('\n')

        loss, acc = float(loss), float(acc)*100
        # print(f"Train Loss = {loss}\tAcc = {acc}\tEps = {eps}")
        train_losses.append(loss)
        train_accs.append(acc)


        test_line = final_lines[2*e+1]
        loss_part = test_line.split('Loss: ')[1]
        loss, acc_part = loss_part.split('Acc@1:')
        acc, eps_part = acc_part.split('\n')

        loss, acc = float(loss), float(acc)*100
        # print(f"Test Loss = {loss}\tAcc = {acc}")
        test_losses.append(loss)
        test_accs.append(acc)

    return train_losses, train_accs, test_losses, test_accs

def parse_results(filepath, num_epochs):
    print(f"Parsing {filepath}...")

    epochs = [e for e in range(1, num_epochs+1)]

    epsilons = []

    test_losses = []
    train_losses = []

    test_accs = []
    train_accs = []

    lines = []
    with open(filepath, "r") as file:
        lines = file.readlines() 

    final_lines = []
    for line in lines:
        if 'Train' in line or 'Test' in line:
            final_lines.append(line)
        
    assert len(final_lines) == 2*num_epochs 

    for e in range(num_epochs):
        train_line = final_lines[2*e]
        loss_part = train_line.split('Loss: ')[1]
        loss, acc_part = loss_part.split('Acc@1:')
        acc, eps_part = acc_part.split('(ε =')
        eps = eps_part.split(', δ = ')[0]

        loss, acc, eps = float(loss), float(acc)*100, float(eps)
        # print(f"Train Loss = {loss}\tAcc = {acc}\tEps = {eps}")
        train_losses.append(loss)
        train_accs.append(acc)
        epsilons.append(eps)


        test_line = final_lines[2*e+1]
        loss_part = test_line.split('Loss: ')[1]
        loss, acc_part = loss_part.split('Acc@1:')
        acc, eps_part = acc_part.split('\n')

        loss, acc = float(loss), float(acc)*100
        # print(f"Test Loss = {loss}\tAcc = {acc}")
        test_losses.append(loss)
        test_accs.append(acc)

    return train_losses, train_accs, test_losses, test_accs, epsilons


def plot_graphs_for_exp(train_losses, train_accs, test_losses, test_accs, epsilons):
    num_epochs = len(train_losses)
    x_points = np.array([e+1 for e in range(num_epochs)])

    fig, (loss_canvas, acc_canvas) = plt.subplots(2, 1)

    loss_canvas.set(xlabel='Epoch', ylabel='Loss')
    loss_canvas.plot(x_points, np.array(train_losses))
    loss_canvas.plot(x_points, np.array(test_losses))
    loss_canvas.legend(['Train','Test'])


    acc_canvas.set(xlabel='Epoch', ylabel='Accuracy (%)')
    acc_canvas.plot(x_points, np.array(train_accs))
    acc_canvas.plot(x_points, np.array(test_accs))
    acc_canvas.legend(['Train','Test'])

    fig.savefig("exp.png")


def plot_train_graphs_for_exp(num_epochs, train_losses_dict, train_accs_dict, epsilons_dict, c):
    x_points = np.array([e+1 for e in range(num_epochs)])

    fig1, (loss_canvas, acc_canvas) = plt.subplots(2, 1)
    fig2, eps_canvas = plt.subplots()

    legend = []
    for exp in train_losses_dict.keys():
        train_losses = train_losses_dict[exp]
        train_accs = train_accs_dict[exp]
        epsilons = epsilons_dict[exp]

        loss_canvas.plot(x_points, train_losses)
        acc_canvas.plot(x_points, train_accs)
        eps_canvas.plot(x_points, epsilons)
        legend.append(exp)

    loss_canvas.xaxis.set_label_text("Epochs")
    loss_canvas.yaxis.set_label_text("Train Loss")
    loss_canvas.legend(legend, prop={'size': 6})

    acc_canvas.xaxis.set_label_text("Epochs")
    acc_canvas.yaxis.set_label_text("Train Acc (%)")
    acc_canvas.legend(legend, prop={'size': 6})

    eps_canvas.xaxis.set_label_text("Epochs")
    eps_canvas.yaxis.set_label_text("Epsilon")
    eps_canvas.legend(legend, prop={'size': 6})

    loss_canvas.set_title("Train Statistics for Batch Size 4000, LR 0.2")
    fig1.savefig(f"./cifar10_results/b4000_c6/train_exp_orca_b4000_c{c}_withdouble.png")

    eps_canvas.set_title("Privacy Exp. for Batch Size 4000 (Delta = 10^-5)")
    fig2.savefig(f"./cifar10_results/b4000_c6/train_exp_orca_b4000_c{c}_privacy.png")


def plot_test_graphs_for_exp(num_epochs, test_losses_dict, test_accs_dict, epsilons_dict, c):
    x_points = np.array([e+1 for e in range(num_epochs)])

    fig1, (loss_canvas, acc_canvas) = plt.subplots(2, 1)

    legend = []
    for exp in test_losses_dict.keys():
        test_losses = test_losses_dict[exp]
        test_accs = test_accs_dict[exp]
        epsilons = epsilons_dict[exp]

        loss_canvas.plot(x_points, test_losses)
        acc_canvas.plot(x_points, test_accs)
        legend.append(exp)

    loss_canvas.xaxis.set_label_text("Epochs")
    loss_canvas.yaxis.set_label_text("Test Loss")
    loss_canvas.legend(legend, prop={'size': 6})

    acc_canvas.xaxis.set_label_text("Epochs")
    acc_canvas.yaxis.set_label_text("Test Acc (%)")
    acc_canvas.legend(legend, prop={'size': 6})

    loss_canvas.set_title("Test Statistics for Batch Size 4000")
    fig1.savefig(f"./cifar10_results/b4000_c6/test_exp_orca_b4000_c{c}_withdouble.png")



num_epochs = 10

def init_dicts():
    train_losses_dict = {}
    train_accs_dict = {}
    test_losses_dict = {}
    test_accs_dict = {}
    epsilons_dict = {}

    return train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict

def form_result_dicts_for_exps(c, sigmas, mode):
    for sigma in sigmas:
        filepath = f"./cifar10_results/b4000_c6/orca_4000_c{c}_s{sigma}_lr0.2{'_'+mode if mode == 'double' else ''}.txt"
        train_losses, train_accs, test_losses, test_accs, epsilons = parse_results(filepath, num_epochs)

        exp_config = f"c = {c} sigma = {sigma} {'double' if mode == 'double' else ''}"
        train_losses_dict[exp_config] = train_losses
        train_accs_dict[exp_config] = train_accs
        test_losses_dict[exp_config] = test_losses
        test_accs_dict[exp_config] = test_accs
        epsilons_dict[exp_config] = epsilons

    # vanilla results for benchmarking
    filepath = f'./cifar10_results/orca_b1000_lr0.05_vanilla.txt'
    train_losses, train_accs, test_losses, test_accs = parse_results_vanilla(filepath, num_epochs)

    exp_config = f"vanilla"
    train_losses_dict[exp_config] = train_losses
    train_accs_dict[exp_config] = train_accs
    test_losses_dict[exp_config] = test_losses
    test_accs_dict[exp_config] = test_accs
    epsilons_dict[exp_config] = [0]*num_epochs

    return train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict


if __name__ == "__main__":
    c = 6
    sigmas = [1, 2, 3]
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = init_dicts()
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "single")
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "double")    
    plot_train_graphs_for_exp(num_epochs, train_losses_dict, train_accs_dict, epsilons_dict, c)
    plot_test_graphs_for_exp(num_epochs, test_losses_dict, test_accs_dict, epsilons_dict, c)

    '''
    c = 5
    sigmas = [2, 3]
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = init_dicts()
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "single")
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "double")    
    plot_train_graphs_for_exp(num_epochs, train_losses_dict, train_accs_dict, epsilons_dict, c)
    plot_test_graphs_for_exp(num_epochs, test_losses_dict, test_accs_dict, epsilons_dict, c)


    c = 6
    sigmas = [2, 3]
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = init_dicts()
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "single")
    train_losses_dict, train_accs_dict, test_losses_dict, test_accs_dict, epsilons_dict = form_result_dicts_for_exps(c, sigmas, "double")
    plot_train_graphs_for_exp(100, train_losses_dict, train_accs_dict, epsilons_dict, c)
    plot_test_graphs_for_exp(num_epochs, test_losses_dict, test_accs_dict, epsilons_dict, c)
    '''