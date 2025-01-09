import argparse
import os
import torch
import numpy as np
import random
import network, loss
import datetime
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from classlibrary import classLibrary, DealDataset
from torch.utils.data import DataLoader
from reprog import ReprogrammableLayer, ReproWrapper, map_target_to_source_classes


cl = classLibrary()

def op_copy(optimizer):
    # Save the initial learning rate of each parameter group
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    # Compute the learning rate decay factor
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    
    # Update each parameter group's learning rate and other hyperparameters
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    
    return optimizer

def cal_acc(loader, netF, netB, netC, device="cpu", flag=False):
    # Initialize flag to start collecting predictions and labels
    start_test = True
    
    # Disable gradient calculations for evaluation
    with torch.no_grad():
        iter_test = iter(loader)
        
        # Iterate through the data loader
        for i in range(len(loader)):
            data = next(iter_test)
            
            # Split data into inputs and labels
            inputs = data[0]
            labels = data[1]

            # Move inputs and labels to the specified device (CPU/GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass through the network
            outputs = netC(netB(netF(inputs.float())))

            # If first batch, initialize storage for outputs and labels
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                # Append predictions and labels to the storage
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    # Apply softmax to compute probabilities from logits
    all_output = torch.nn.Softmax(dim=1)(all_output)
    
    # Get predicted class by finding the index of the maximum probability
    _, predict = torch.max(all_output, 1)
    
    # Calculate accuracy by comparing predictions with ground truth labels
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    # Compute the mean entropy of the predictions
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        # Compute the confusion matrix if flag is set
        matrix = confusion_matrix(all_label.numpy(), torch.squeeze(predict).float().numpy())
        
        # Calculate accuracy per class from the confusion matrix
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        
        # Format class-wise accuracies for logging
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        
        return aacc, acc
    else:
        # Return overall accuracy and mean entropy
        return accuracy * 100, mean_ent
    
def cal_acc_reprog(loader, reprog_net, device="cpu", flag=False):
    start_test = True
    reprog_net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Forward pass with reprogramming
            outputs = reprog_net(inputs)
            
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    # Same post-processing as original cal_acc
    all_output = torch.nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(predict == all_label).item() / float(all_label.size(0)) * 100.0

    # You can also compute confusion matrix if needed
    return accuracy, None

# Prepares data loaders for training and testing source model.
def data_load(args, test):
    # Prepare data dictionaries
    dsets = {}
    dset_loaders = {}
    
    # Batch size for training
    train_bs = args.batch_size

    # Dataset paths
    file_dirs_sor = args.s_dset_path  # Source dataset path
    file_dirs_test = args.test_dset_path  # Target dataset path
    
    # Data preparation settings
    time_len = args.time_len  # Fixed time length for sequences
    num_class = args.classes  # Number of classes
    
    # Load source dataset
    dsets["source"] = cl.data_conca(file_dirs_sor, time_len, [5], num_class, args.num_dataset)

    # Split the source dataset into training and validation if required
    if args.trte == "val":
        dsize = dsets["source"].shape[0]
        tr_size = int(0.1 * dsize)  # Total size per class for training and validation
        tr_size_class = int(0.9 * tr_size)  # Training size per class

        source_tr = np.zeros((1, args.time_len, 51))  # Temporary storage for training data
        source_te = np.zeros((1, args.time_len, 51))  # Temporary storage for testing data

        for i in range(num_class):
            # Split each class into training and validation
            source_tr_class = dsets["source"][tr_size * i : (tr_size * i + tr_size_class), :, :]
            source_te_class = dsets["source"][(tr_size * i + tr_size_class) : tr_size * (i + 1), :, :]
            source_tr = np.concatenate((source_tr, source_tr_class), axis=0)
            source_te = np.concatenate((source_te, source_te_class), axis=0)

        # Remove the first dummy entry
        dsets["source_tr"] = source_tr[1:source_tr.shape[0], :, :]
        dsets["source_te"] = source_te[1:source_te.shape[0], :, :]
    else:
        raise ValueError("Non-validation splits are not implemented yet!")

    # Prepare datasets for training, testing, and validation
    dsets["source_tr"] = DealDataset(dsets["source_tr"], 4)
    dsets["source_te"] = DealDataset(dsets["source_te"], 4)
    dsets["test"] = cl.data_conca(file_dirs_test, time_len, [5], num_class, args.num_dataset)
    dsets["test"] = DealDataset(dsets["test"], 4)

    # Shuffle data during training, keep order during testing
    shuffle = test == 0

    # Create DataLoaders for training and testing
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"], batch_size=train_bs, shuffle=shuffle, num_workers=args.worker, drop_last=False
    )
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"], batch_size=train_bs, shuffle=shuffle, num_workers=args.worker, drop_last=False
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=train_bs, shuffle=shuffle, num_workers=args.worker, drop_last=False
    )

    return dset_loaders

# Trains source model
def train_source(args):
    # Load dataset loaders
    dset_loaders = data_load(args, 0)

    # Set up the network components
    print("Initializing networks...")
    netF = network.MyAlexNet(resnet_name=args.net).to(args.device)

    dummy_input = torch.zeros(1, 1, args.time_len, 51).to(args.device)  # Shape: (batch_size=1, channels=1, time_len, features)
    dummy_output = netF(dummy_input)
    netF.in_features = dummy_output.numel()  # Total flattened features

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).to(args.device)
    netC = network.feat_classifier(type=args.layer, class_num=args.classes, bottleneck_dim=args.bottleneck).to(args.device)

    # Prepare optimizer with different learning rates for each network component
    print("Setting up optimizer...")
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-4)
    optimizer = op_copy(optimizer)

    # Initialize training variables
    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    ACC = []

    # Set networks to training mode
    netF.train()
    netB.train()
    netC.train()

    # Initialize the iterator for the source training loader
    iter_source = iter(dset_loaders["source_tr"])

    # Training loop
    print("Starting training...")
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except StopIteration:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        # Adjust learning rate using a scheduler
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # Move data to the appropriate device
        inputs_source, labels_source = inputs_source.to(args.device), labels_source.to(args.device)

        # Forward pass through the network
        outputs_source = netC(netB(netF(inputs_source.float())))

        # Compute loss
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.classes, epsilon=args.smooth)(
            outputs_source, labels_source.long()
        )

        # Backward pass and optimizer step
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # Periodically evaluate and log performance
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            # Evaluate on source test data
            acc_s_te, acc_list = cal_acc(dset_loaders["source_te"], netF, netB, netC, flag=False)
            log_str = f"Task: {args.name_src}, Iter: {iter_num}/{max_iter}; Accuracy = {acc_s_te:.2f}%"
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            # Track best performance and save models
            ACC.append(acc_s_te)
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            # Switch back to training mode
            netF.train()
            netB.train()
            netC.train()

    # Save the best model checkpoints
    print("Saving best model...")
    torch.save(best_netF, os.path.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, os.path.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, os.path.join(args.output_dir_src, "source_C.pt"))

    print("Training complete.")
    return netF, netB, netC, dset_loaders

# Function to test the raw source model on the target dataset
def test_source(args, dset_loaders, netF, netB, netC):
    # Set networks to evaluation mode
    netF.eval()
    netB.eval()
    netC.eval()

    print("Testing source model on target dataset...")
    
    # Evaluate accuracy on target test dataset
    accuracy, _ = cal_acc(dset_loaders["test"], netF, netB, netC, device=args.device, flag=False)
    
    # Log the result
    log_str = f"Raw Source Model Accuracy on Target Dataset: {accuracy:.2f}%\n"
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

# Reprogram the source model using target data.
def reprogram_source(netF, netB, netC, dset_loaders, args):
    print("Initializing Reprogramming...")

    # 1) Freeze the source model
    for param in netF.parameters():
        param.requires_grad = False
    for param in netB.parameters():
        param.requires_grad = False
    for param in netC.parameters():
        param.requires_grad = False

    # 2) The input shape and reprogram layer
    #    Suppose your source model expects time_len=60, features=50 (example),
    #    and your target length = args.targ_len. 
    #    We pass 'args.tmp' as the original source time_len from training.
    source_time_len = args.tmp      # e.g. 60
    target_time_len = args.targ_len # e.g. 10
    input_shape = (args.batch_size, target_time_len, 50)  
    reprog_layer = ReprogrammableLayer(
        input_shape, 
        source_time_len,   # the number of time steps the source net expects
        target_time_len,
        device=args.device
    ).to(args.device)

    # 3) Build a wrapper that includes reprogramming
    reprog_net = ReproWrapper(reprog_layer, netF, netB, netC).to(args.device)

    # 4) Set up optimizer
    optimizer = torch.optim.Adam(reprog_layer.parameters(), lr=args.lr)

    # 5) Total training iterations
    max_iter = args.target_max_epoch * len(dset_loaders["test"])
    interval_iter = max_iter // 10
    iter_num = 0

    # 6) Map target classes -> source classes 
    #    (assuming same # of classes => identity mapping)
    target_to_source_map = map_target_to_source_classes(args.classes, args.classes)

    # 7) Train
    reprog_net.train()
    acc_init = 0  # track best accuracy
    print("Starting reprogramming training...")

    while iter_num < max_iter:
        for inputs_target, labels_target in dset_loaders["test"]:
            iter_num += 1

            # Move to device
            inputs_target = inputs_target.to(args.device)
            labels_target = labels_target.to(args.device)

            # Forward
            outputs_reprog = reprog_net(inputs_target)

            # Map target -> source labels
            mapped_labels = target_to_source_map[labels_target.long()]

            # Loss
            loss = torch.nn.CrossEntropyLoss()(outputs_reprog, mapped_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate & log every interval or at final iteration
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                with torch.no_grad():
                    reprog_net.eval()
                    # Compute accuracy on target data with reprogramming
                    accuracy, _ = cal_acc_reprog(dset_loaders["test"], reprog_net, args.device)
                    
                    log_str = f"Task: {args.name_src}, Iter: {iter_num}/{max_iter}; Accuracy = {accuracy:.2f}%"
                    
                    # Output to console
                    print(log_str + '\n')
                    # Output to log file
                    args.out_file.write(log_str + '\n')
                    args.out_file.flush()

                    # Track best accuracy
                    if accuracy >= acc_init:
                        acc_init = accuracy
                        # Optionally save the best reprogramming layer
                        # torch.save(reprog_layer.state_dict(), os.path.join(args.output_dir_src, "best_reprog_layer.pt"))

                    reprog_net.train()

            if iter_num >= max_iter:
                break

    # Save final reprogram parameters
    print("Saving reprogramming parameters...")
    torch.save(reprog_layer.state_dict(), os.path.join(args.output_dir_src, "reprog_layer.pt"))
    print("Reprogramming complete.")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=1, help="source mode [1, 6]")
    parser.add_argument("--t", type=int, default=3, help="target mode [1, 6]")
    parser.add_argument("--classes", type=int, default=10, help="Num classes [2, 21]")
    parser.add_argument("--num_dataset", type=int, default=800, help="Num classes [2, 21]")
    parser.add_argument("--time_len", type=int, default=60, help="Num classes [2, 21]")
    parser.add_argument("--targ_len", type=int, default=20, help="Num classes [2, 21]")
    parser.add_argument("--max_epoch", type=int, default=5, help="max iterations")
    parser.add_argument("--target_max_epoch", type=int, default=50, help="max iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--worker", type=int, default=0, help="number of workers")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--net", type=str, default="vgg16", help="vgg16, resnet50, resnet101")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    args = parser.parse_args()
    args.s -= 1
    args.t -= 1

    # Define Mode Names
    names = ["mode1", "mode2", "mode3", "mode4", "mode5", "mode6"]
    class_num = args.classes

    # Set Random Seed for Reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Set CUDA seed only if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Determine the available device (CPU or CUDA-enabled GPU)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Define Path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Data folder: {current_dir}/TE_process_data/TE_new/
    data_folder = os.path.join(current_dir, "TE_process_data", "TE_new")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    # Dataset paths for source and target modes
    args.s_dset_path = os.path.join(data_folder, f"{names[args.s]}_new/")
    args.test_dset_path = os.path.join(data_folder, f"{names[args.t]}_new/")

    # Output folder: {current_dir}/output
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%H%M_%d%m")  # Format: HHMM_DDMM
    output_folder = os.path.join(current_dir, args.output)
    folder_name = f"{timestamp}_{names[args.s]}_{names[args.t]}_c{args.classes}_t{args.time_len}"  # Format: HHMM_DDMM_src_tr_classes
    args.output_dir_src = os.path.join(output_folder, folder_name)
    args.name_src = folder_name  # Use this name for logging and file organization

    # Ensure the output directory exists
    os.makedirs(args.output_dir_src, exist_ok=True)

    # Set Up Log File
    args.out_file = open(os.path.join(args.output_dir_src, "log.txt"), "w")
    args.out_file.write(str(args) + "\n")  # Record the arguments used
    args.out_file.flush()

    # Train the Source Model
    netF, netB, netC, dset_loaders = train_source(args)

    # Set Up Test Log File
    args.out_file = open(os.path.join(args.output_dir_src, "log_test.txt"), "w")
    args.out_file.write(str(args) + "\n")  # Record the arguments used
    args.out_file.flush()

    # Test Raw Source Model on Target
    test_source(args, dset_loaders, netF, netB, netC)

    # Switch time window length
    args.tmp = args.time_len
    args.time_len = args.targ_len
    dset_loaders = data_load(args, 0)

    # Perform Reprogramming
    reprogram_source(netF, netB, netC, dset_loaders, args)
