# -*- coding: utf-8 -*-
"""
Author -- Favour Akpasi
Main file of Depixilation project.
"""

import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.stacking import stack_with_padding
from utils.image_preparation import ImagePreprocessing
import torch
from utils.plotting import plot
from utils.architecture import SimpleCNN
import warnings
import numpy as np
from tqdm import tqdm


def evaluate_model(model: torch.nn.Module, test_set, loader: torch.utils.data.DataLoader, loss_fn,
                   device: torch.device):
    model.eval()
    predictions = []

    test_set_file = os.path.abspath(r'./test_set/test_set.pkl')
    with open(test_set_file, "rb") as f:
        test_set = pickle.load(f)

    pixelated_images = test_set["pixelated_images"]
    known_arrays = test_set["known_arrays"]

    with torch.no_grad():
        for i in range(len(pixelated_images)):
            pixelated_image = pixelated_images[i]
            known_array = known_arrays[i]

            pixelated_image_tensor = torch.tensor(pixelated_image).float().to(device)
            known_array_tensor = torch.tensor(known_array).float().to(device)

            input_tensor = torch.cat((pixelated_image_tensor, known_array_tensor))

            output = model(input_tensor)

            # Apply boolean mask to obtain predicted pixel values
            predicted_values = output.squeeze(0) * known_array_tensor

            known_array = torch.tensor(known_array)
            # Flatten the predicted values using the boolean mask
            predicted_values_flat = predicted_values[~known_array]

            # Convert predicted values to 1D NumPy array of type np.uint8
            predicted_values_flat = np.uint8(predicted_values_flat.cpu().numpy())

            predictions.append(predicted_values_flat)

    model.train()
    return predictions


def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50,
        device: str = "cuda"
):
    """Main function that takes hyperparameters and performs training and
    evaluation of model"""
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    dataset = ImagePreprocessing(
        image_dir=os.path.abspath(r'./test'),
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )
    training_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5)))
    )
    test_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset))
    )

    # Create data loaders
    train_loader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=stack_with_padding)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # Define a TensorBoard summary writer that writes to directory
    results_path = os.path.abspath(r'./results')
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    write_stats_at = 1  # Write status to TensorBoard every x updates
    plot_at = 1  # Plot every x updates
    validate_at = 5000  # Evaluate model on validation set and check for new best model every x updates
    update = 0  # Current update counter
    best_validation_loss = np.inf  # Best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:
        for data in train_loader:
            # Get next samples
            pixelated_images, known_arrays, targets, _ = data
            pixelated_images = torch.tensor(pixelated_images).to(device)
            known_arrays = torch.tensor(known_arrays).to(device)
            targets = torch.tensor(targets)
            targets = torch.squeeze(targets, dim=1).to(device)

            input_tensor = torch.cat((pixelated_images, known_arrays))

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = net(input_tensor.float())

            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss = loss.float()
            loss.backward()
            optimizer.step()

            # Write current training status
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)

            # Plot output
            if (update + 1) % plot_at == 0:
                targets = torch.unsqueeze(targets, dim=1)
                pixelated_images = torch.unsqueeze(pixelated_images, dim=1)
                outputs = torch.unsqueeze(outputs, dim=1)
                plot(pixelated_images.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                     outputs.detach().cpu().numpy(),
                     plot_path, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device, test_set=test_set)
                writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            # Increment update counter, exit if maximum number of updates is
            # reached. Here, we could apply some early stopping heuristic and
            # also exit if its stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, test_set, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, test_set, loader=val_loader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, test_set, loader=test_loader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, nargs="?", default="utils/config_file.json",
                        help="Path to the configuration file")
    args = parser.parse_args()
    print(f"Configuration file path: {args.config_file}")

    with open(args.config_file) as cf:
        config = json.load(cf)
    print(f"Loaded config: {config}")
    main(**config)
