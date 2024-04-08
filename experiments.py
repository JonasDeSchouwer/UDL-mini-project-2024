import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from datetime import datetime
from datasets import create_dataset, Dataset
from utils import id_to_idxs, bay_train_writer_layout
from models import VanillaNN, BayesianNN
from coreset import MultiTaskDataContainer, create_coreset_method


class TrainParams:
    pass

class VanTrainParams(TrainParams):
    def __init__(self):
        # all are taken from paper implementation
        self.num_epochs = 120
        # full batch size
        self.learning_rate = 0.001
        self.display_epoch = 50
        # cross entropy loss function (to compute NLL)
        # Adam optimizer

class BayTrainParams(TrainParams):
    def __init__(self):
        # all are taken from paper implementation
        self.num_epochs = 120
        # full batch size
        self.learning_rate = 0.001
        self.display_epoch = 50
        self.num_train_samples = 10 # the number of times we resample the weights per feedforward pass (i.e. per mini batch)
        self.num_pred_samples = 100 # the number of times we resample the weights per feedforward pass (i.e. per mini batch)
        # cross entropy loss function (to compute NLL)
        # Adam optimizer


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, task_idxs):
    """
    Evaluate the model accuracy on the given data
    """

    preds = torch.argmax(model(x, task_idxs), dim=1)
    labels = torch.argmax(y, dim=1)
    acc = (preds == labels).float().mean()
    return acc.item()

def evaluate_sampled(model: BayesianNN, x: torch.Tensor, y: torch.Tensor, task_idxs, num_samples=100):
    """
    Evaluate the model accuracy on the given data, when sampling the weights and biases of the Bayesian network, like in the paper implementation
    """
    outputs = torch.zeros((num_samples, *y.shape), dtype=x.dtype, device=x.device)
    for i in range(num_samples):
        model.resample_eps()
        outputs[i] = model(x, task_idxs)
    preds = torch.argmax(outputs.mean(dim=0), dim=1)
    labels = torch.argmax(y, dim=1)
    acc = (preds == labels).float().mean()
    return acc.item()

def evaluate_multi_task(model: nn.Module, data: MultiTaskDataContainer):
    """
    Evaluate the model accuracy on the given data
    """
    num_correct = 0
    num_total = 0
    for task_idxs, x, y in data:
        num_correct += len(x) * evaluate(model, x, y, task_idxs)
        num_total += len(x)
    return num_correct / num_total

def evaluate_multi_task_sampled(model: BayesianNN, data: MultiTaskDataContainer, num_samples=100):
    """
    Evaluate the model accuracy on the given data, when sampling the weights and biases of the Bayesian network, like in the paper implementation
    """
    num_correct = 0
    num_total = 0
    for task_idxs, x, y in data:
        num_correct += len(x) * evaluate_sampled(model, x, y, task_idxs, num_samples)
        num_total += len(x)
    return num_correct / num_total


def train_vanilla(model: VanillaNN, params: VanTrainParams, x_train, y_train, x_test, y_test, task_idxs, verbose=True):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss() # computes the NLL after softmax to outputs

    for epoch in range(params.num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train, task_idxs)    # train with full batch, like in paper implementation
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if verbose and epoch % params.display_epoch == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Test accuracy: {evaluate(model, x_test, y_test, task_idxs)}")

def train_bayesian(model: BayesianNN, prior_model: BayesianNN, params: BayTrainParams, train_data: MultiTaskDataContainer, x_test, y_test, loop_task_id, shared_lr=None, verbose=True, writer_path=None):

    if verbose:
        print("Len training data:", train_data.lens())

    if len(train_data) == 0:
        print("No training data, skipping training")
        return
    
    if shared_lr is None:
        shared_lr = params.learning_rate

    # Initialize tensorboard logger
    writer = None
    if writer_path is not None:
        writer = SummaryWriter(writer_path)
        writer.add_custom_scalars(bay_train_writer_layout(loop_task_id))
    
    optimizer = torch.optim.Adam([
        {'params': model.shared_parameters(), 'lr': shared_lr},
        {'params': model.task_specific_parameters(), 'lr': params.learning_rate}
    ], lr=params.learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction='sum') # computes the NLL after softmax to outputs

    for epoch in range(params.num_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute the negative log likelihood term
        nll_term = 0
        for _ in range(params.num_train_samples):
            model.resample_eps()
            for task_idxs, x_train, y_train in train_data:
                y_pred = model(x_train, task_idxs)    # train with full batch, like in paper implementation
            nll_term += loss_fn(y_pred, y_train)
        # Divide by number of training samples, because we approximate the expected NLL with a Monte Carlo estimate
        # Divide by the length of the training data, because we do the same to the KL term (and this makes the algorithm invariant to the size of the training set)
        nll_term /= params.num_train_samples * len(train_data)

        # Compute the KL term
        # Note that both the KL term and the negative log likelihood term are divided by the number of training samples compared to the paper (but as in the paper implementation)
        const_term, log_std_term, mu_diff_term, std_quotient_term = model.KL(prior_model, components=True)
        const_term /= len(train_data)
        log_std_term /= len(train_data)
        mu_diff_term /= len(train_data)
        std_quotient_term /= len(train_data)

        kl_term = const_term + log_std_term + mu_diff_term + std_quotient_term

        loss = nll_term + kl_term
        loss.backward()

        optimizer.step()

        # --- EVERYTHING BELOW THIS IS FOR LOGGING PURPOSES ---

        if writer is not None:
            try:
                writer.add_scalar(f"task{loop_task_id}/loss/total", loss.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/NLL", nll_term.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/KL", kl_term.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/KL/const_term", const_term.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/KL/log_std_term", log_std_term.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/KL/mu_diff_term", mu_diff_term.item(), epoch)
                writer.add_scalar(f"task{loop_task_id}/loss/KL/std_quotient_term", std_quotient_term.item(), epoch)
            except Exception as e:
                print("!!! Warning: ", e)

        if epoch % params.display_epoch == 0 and (verbose or writer is not None):
            model.zero_eps()
            train_acc = evaluate_multi_task(model, train_data)
            test_acc = evaluate(model, x_test, y_test, id_to_idxs(loop_task_id))

            if writer is not None:
                try:
                    writer.add_scalar(f"task{loop_task_id}/acc/train", train_acc, epoch)
                    writer.add_scalar(f"task{loop_task_id}/acc/test", test_acc, epoch)
                except Exception as e:
                    print("!!! Warning: ", e)

            if verbose:
                print(f"Epoch {epoch}, Loss: {round(loss.item(), 4)}, Train Accuracy: {round(train_acc, 3)}, Test Accuracy: {round(test_acc, 3)}")
    
    if writer is not None:
        writer.close()


def run_vcl_experiment(model_params, dataset_params, coreset_params, van_train_params, bay_train_params, lr_decay=1, num_runs=5, study_name=None, use_tensorboard=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)

    # Initialize dataset
    dataset = create_dataset(dataset_params["type"]).to(device)

    if study_name is None:
        study_name = datetime.strftime(datetime.now(), "%m.%d-%H.%M.%S")

    # 3D tensor that will hold the accuracies for each task
    # Format: accuracies[run_id][eval_task_id][loop_task_id]
    accuracies = torch.zeros((num_runs, dataset.num_tasks, dataset.num_tasks), dtype=torch.float32)
    
    for run_id in range(num_runs):
        print(f"\n--- STARTING RUN {run_id} ---")
        torch.manual_seed(run_id)   # for reproducibility

        # Initialize models
        van_nn = VanillaNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        bay_nn = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        prior_model = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        prior_model.initialize_to_prior(prior_mu=0, prior_sigma=1)
        pred_model = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)

        # Initialize coreset method
        coreset_method = create_coreset_method(dataset, coreset_params["type"], coreset_params["size"])

        # Initialize shared learning rate (will be decayed after each task)
        shared_lr = bay_train_params.learning_rate

        # Loop over the tasks
        for loop_task_id in range(dataset.num_tasks):
            # The training data comes from $\tilde{D}_t$
            train_data, coreset = next(coreset_method)
            assert loop_task_id == coreset_method.task_id
            # Get test data and task indices
            x_test, y_test = dataset.get_test_data(loop_task_id)
            task_idxs = dataset.get_task_idxs(loop_task_id)

            # Train vanilla model on task 0 to initialize the Bayesian model (note: we train only 1/5th of the last layer)
            if loop_task_id == 0:
                print("Training vanilla model on task 0 to initialize the Bayesian model...")
                x_train, y_train = train_data[loop_task_id]
                train_vanilla(van_nn, van_train_params, x_train, y_train, x_test, y_test, task_idxs, verbose=True)
                bay_nn.initialize_using_vanilla(van_nn, prior_mu=0, prior_sigma=1)    # like in paper implementation


            # --- FIRST KL MINIMIZATION LOOP ---

            # Initialize prior to previous iteration's model
            if loop_task_id > 0:
                prior_model.load_state_dict(bay_nn.state_dict())

            print(f"Training Bayesian model on task {loop_task_id} training data...")
            print(f"Shared learning rate: {shared_lr}")
            writer_path = f"{study_name}/run{run_id}" if use_tensorboard else None
            train_bayesian(bay_nn, prior_model, bay_train_params, train_data, x_test, y_test, loop_task_id=loop_task_id, shared_lr=shared_lr, writer_path=writer_path, verbose=True)


            # --- SECOND KL MINIMIZATION LOOP ---
            # Important note: unlike the paper implementation, we first train the model on the full coreset and then evaluate it on all test datasets
            # The paper implementation uses an interleaved evaluation loop (train on coreset task 0, evaluate on test data task 0, train on coreset task 1, evaluate on test data task 1, ...) which gives an unfair advantage

            # Initialize prior and prediction model to previous iteration's model
            prior_model.load_state_dict(bay_nn.state_dict())
            pred_model.load_state_dict(bay_nn.state_dict())

            print(f"Training prediction model on task {loop_task_id} Coreset...")
            writer_path = f"{study_name}/run{run_id}-coreset" if use_tensorboard else None
            train_bayesian(pred_model, prior_model, bay_train_params, coreset, x_test, y_test, loop_task_id=loop_task_id, shared_lr=shared_lr, writer_path=writer_path, verbose=True)

            # After training: evaluate model accuracy on every task seen so far
            for eval_task_id in range(loop_task_id + 1):
                pred_model.zero_eps()
                x_test, y_test = dataset.get_test_data(eval_task_id)
                task_idxs = id_to_idxs(eval_task_id)
                acc = evaluate_sampled(pred_model, x_test, y_test, task_idxs, num_samples=bay_train_params.num_pred_samples)
                accuracies[run_id][eval_task_id][loop_task_id] = acc
                print(f"Task {eval_task_id} accuracy: {round(acc, 3)}")

            # Save this run's accuracies
            Path(f"{study_name}/run{run_id}").mkdir(parents=True, exist_ok=True)
            torch.save(accuracies[run_id], f"{study_name}/run{run_id}/accuracies.pt")

            # Decay the shared learning rate
            shared_lr *= lr_decay

    torch.save(accuracies, f"{study_name}/accuracies.pt")


def run_vcl_experiment_interleaved_testing(model_params, dataset_params, coreset_params, van_train_params, bay_train_params, num_runs=5, study_name=None, use_tensorboard=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)

    # Initialize dataset
    dataset = create_dataset(dataset_params["type"]).to(device)

    if study_name is None:
        study_name = datetime.strftime(datetime.now(), "%m.%d-%H.%M.%S")

    # 3D tensor that will hold the accuracies for each task
    # Format: accuracies[run_id][eval_task_id][loop_task_id]
    accuracies = torch.zeros((num_runs, dataset.num_tasks, dataset.num_tasks), dtype=torch.float32)
    
    for run_id in range(num_runs):
        print(f"\n--- STARTING RUN {run_id} ---")
        torch.manual_seed(run_id)   # for reproducibility

        # Initialize models
        van_nn = VanillaNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        bay_nn = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        prior_model = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)
        prior_model.initialize_to_prior(prior_mu=0, prior_sigma=1)
        pred_model = BayesianNN(
            num_hidden_layers=model_params["num_hidden_layers"], input_dim=model_params["in_dim"], hidden_dim=model_params["hidden_dim"], output_dim=model_params["out_dim"]
        ).to(device)

        # Initialize coreset method
        coreset_method = create_coreset_method(dataset, coreset_params["type"], coreset_params["size"])

        # Loop over the tasks
        for loop_task_id in range(dataset.num_tasks):
            # The training data comes from $\tilde{D}_t$
            train_data, coreset = next(coreset_method)
            assert loop_task_id == coreset_method.task_id
            # Get test data and task indices
            x_test, y_test = dataset.get_test_data(loop_task_id)
            task_idxs = dataset.get_task_idxs(loop_task_id)

            # Train vanilla model on task 0 to initialize the Bayesian model (note: we train only 1/5th of the last layer)
            if loop_task_id == 0:
                print("Training vanilla model on task 0 to initialize the Bayesian model...")
                x_train, y_train = train_data[loop_task_id]
                train_vanilla(van_nn, van_train_params, x_train, y_train, x_test, y_test, task_idxs, verbose=True)
                bay_nn.initialize_using_vanilla(van_nn, prior_mu=0, prior_sigma=1)    # like in paper implementation


            # --- FIRST KL MINIMIZATION LOOP ---

            # Initialize prior to previous iteration's model
            if loop_task_id > 0:
                prior_model.load_state_dict(bay_nn.state_dict())

            print(f"Training Bayesian model on task {loop_task_id} training data...")
            writer_path = f"{study_name}/run{run_id}" if use_tensorboard else None
            train_bayesian(bay_nn, prior_model, bay_train_params, train_data, x_test, y_test, loop_task_id=loop_task_id, writer_path=writer_path, verbose=True)


            # --- SUBSEQUENT KL MINIMIZATION LOOPS ---
            # Like in the paper implementation, we use an interleaved evaluation loop (train on coreset task 0, evaluate on test data task 0, train on coreset task 1, evaluate on test data task 1, ...) which gives an unfair advantage

            # Initialize prior and prediction model to previous iteration's model
            prior_model.load_state_dict(bay_nn.state_dict())
            pred_model.load_state_dict(bay_nn.state_dict())

            for coreset_task_id in range(loop_task_id + 1):
                if coreset_task_id not in coreset.task_ids():
                    print(f"No coreset data for task {coreset_task_id}, skipping...")
                    continue

                # Train the prediction model on the coreset data of task `coreset_task_id`
                # Note: prior model is not updated after any of these training steps
                
                print(f"Training prediction model on task {loop_task_id} Coreset: task {coreset_task_id} data...")
                current_train_data = MultiTaskDataContainer()   # container for the train data that the coreset has for task `coreset_task_id`
                current_train_data[coreset_task_id] = coreset[coreset_task_id]
                x_test, y_test = dataset.get_test_data(coreset_task_id)
                writer_path = f"{study_name}/run{run_id}-coreset{coreset_task_id}" if use_tensorboard else None
                
                train_bayesian(pred_model, prior_model, bay_train_params, current_train_data, x_test, y_test, loop_task_id=loop_task_id, writer_path=writer_path, verbose=True)

                # After training: evaluate model accuracy on the task that was just seen
                pred_model.zero_eps()
                x_test, y_test = dataset.get_test_data(coreset_task_id)
                task_idxs = id_to_idxs(coreset_task_id)
                acc = evaluate_sampled(pred_model, x_test, y_test, task_idxs, num_samples=bay_train_params.num_pred_samples)
                accuracies[run_id][coreset_task_id][loop_task_id] = acc
                print(f"Task {coreset_task_id} accuracy: {round(acc, 3)}")

                # Save this run's accuracies
                Path(f"{study_name}/run{run_id}").mkdir(parents=True, exist_ok=True)
                torch.save(accuracies[run_id], f"{study_name}/run{run_id}/accuracies.pt")

    torch.save(accuracies, f"{study_name}/accuracies.pt")