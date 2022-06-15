import torch
from tqdm import tqdm
import os
from pyifttt.webhook import send_notification
from src.backbones.UTAE.utae import UTAE

from torch.utils.tensorboard import SummaryWriter

def train_model(
    model, 
    optimizer, 
    loss_function, 
    n_epochs, 
    batch_size,
    train_loader, 
    val_loader, 
    device, 
    log_dir,
    save_dir,
    name,
    ):
    """
    Trains the given model for a given number of epochs.
    """
    #TODO: Create log writer
    writer = SummaryWriter(log_dir+'/'+name)

    # Train the model
    for epoch in tqdm(range(n_epochs), desc='Training'):
        # Train the model
        model.train(True)
        # Run the training epoch
        avg_loss = one_epoch(
            model=model, 
            optimizer=optimizer, 
            loss_function=loss_function, 
            data_loader=train_loader, 
            batch_size=batch_size,
            device=device,
            writer=writer,
            )

        # Run the validation epoch 
        model.eval()
        val_loss = one_epoch(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            data_loader=val_loader,
            batch_size=batch_size,
            device=device,
            writer=writer,
            validation=True
        )
        out_string = 'Epoch {}: train loss: {}, val loss: {}'.format(epoch + 1, avg_loss, val_loss)
        print(out_string)
        if os.environ.get('IFTTT_KEY') is not None:
            data = {
                'value1': str(model),
                'value2': out_string
            }
            send_notification('python_notification', data=data)

        save_model(
            model,
            epoch,
            optimizer,
            avg_loss,
            save_dir,
            name+f"_{epoch}"
        )

def one_epoch(
    model, 
    optimizer, 
    loss_function, 
    data_loader, 
    batch_size,
    device,
    writer,
    validation=False
    ):
    """
    Runs one epoch of training.
    """
    # Init tracking variables
    avg_loss = 0.
    running_loss = 0.

    # Get the number of batches
    n_batches = len(data_loader) // batch_size
    if len(data_loader) % batch_size != 0:
        n_batches += 1
    batch_iterator = batch_maker(data_loader, batch_size, n_batches)
    
    description = 'Training epoch' if not validation else 'Validation epoch'
    for i, data in tqdm(
        enumerate(batch_iterator),
        total=n_batches,
        desc=description
    ):
        # Get the inputs and labels
        inputs, labels, time = data
        # Move the batch to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        time = time.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        if not validation:
            if type(model) == UTAE:
                outputs = model(inputs, batch_positions=time)
            else:
                outputs = model(inputs)

            # Compute the loss
            loss = loss_function(outputs, labels.long())
            loss.backward()
            # Update the weights
            optimizer.step()
        else:
            with torch.no_grad():
                if type(model) == UTAE:
                    outputs = model(inputs, batch_positions=time)
                else:
                    outputs = model(inputs)
                # Compute the loss
                loss = loss_function(outputs, labels.long())

        # Update the tracking variables
        running_loss += loss.item()
        avg_loss += loss.item()
        if i % batch_size == batch_size -1:
            last_loss = running_loss / batch_size
            #TODO: Log the loss
            writer.add_scalar('{} loss'.format('Training' if not validation else 'Validation'), last_loss, i+1)
            if not validation: 
                print(' \tbatch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
        
        # Garbage collection
        del inputs, labels, time, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return avg_loss / (i + 1)

def batch_maker(data_set, batch_size, n_batches):
    """
    Creates a generator that yields batches of data.
    """
    
    # Create an iterator for the data
    data_iterator = iter(data_set)

    # Create a generator that yields batches
    for _ in range(n_batches):
        # Create a list to hold the samples in the batch
        single_inputs = []
        single_labels = []
        single_times = []
        # Get the next batch
        for _ in range(batch_size):
            try:
                single_input, single_label, single_time = next(data_iterator)
            except StopIteration:
                break
            # Add the sample to the batch
            single_inputs.append(single_input)
            single_labels.append(single_label)
            single_times.append(single_time)
        
        # Stack the single samples into a batch
        inputs = torch.stack(single_inputs)
        labels = torch.stack(single_labels)
        times = torch.stack(single_times)

        # Garbage collection
        del single_inputs, single_labels, single_times
        # Return the batch
        yield inputs, labels, times

def save_model(model, epoch, optimizer, loss, save_dir, name):

    model_name = name.split('[')[0]
    if not os.path.exists(os.path.join(save_dir, model_name)):
        os.makedirs(os.path.join(save_dir, model_name))

    path = os.path.join(save_dir, model_name, name) + '.md5'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)