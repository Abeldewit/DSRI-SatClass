import torch
from tqdm import tqdm
import os
from pyifttt.webhook import send_notification
from src.backbones.UTAE.utae import UTAE
from datetime import datetime
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, JaccardIndex, ConfusionMatrix

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
    # Initialize the log writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{log_dir}/{name}_{timestamp}')

    # Train the model
    best_vloss = float('inf')
    for epoch_number in tqdm(range(n_epochs), desc='Total Training: '):
        # Initialize the metrics
        acc = Accuracy(num_classes=20, mdmc_average='global').to(device)
        prec = Precision(num_classes=20, mdmc_average='global').to(device)
        rec = Recall(num_classes=20, mdmc_average='global').to(device)
        f1 = F1Score(num_classes=20, mdmc_average='global').to(device)
        jaccard = JaccardIndex(num_classes=20).to(device)
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(
            optimizer, 
            loss_function, 
            model, 
            train_loader, 
            batch_size, 
            device, 
            epoch_number, 
            writer,
        )

        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.
        for i, vdata in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating: '):
            vinputs, vlabels, vtimes = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            vtimes = vtimes.to(device) if isinstance(vtimes, torch.Tensor) else vtimes

            # Make predictions for this batch
            voutputs = model(vinputs) if not isinstance(model, UTAE) else model(vinputs, batch_positions=vtimes)
            vprediction = torch.argmax(voutputs, dim=1)

            # Compute the loss and its gradients
            vloss = loss_function(voutputs, vlabels.long())
            running_vloss += vloss.item()

            # Compute the metrics
            acc(vprediction, vlabels.long())
            prec(vprediction, vlabels.long())
            rec(vprediction, vlabels.long())
            f1(vprediction, vlabels.long())
            jaccard(vprediction, vlabels.long())

            # Garbage collection
            del vinputs, vlabels, vtimes, voutputs, vprediction, vloss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Report the loss
        avg_vloss = running_vloss / (i + 1)
        send_notification('python_notification', data={
            'value1': f'Train loss: {avg_loss:.4f}', 
            'value2': f'Val loss: {avg_vloss:.4f}'
            }
        )

        writer.add_scalars(
            'Training vs. Validation Loss',
            { 'Training': avg_loss, 'Validation': avg_vloss },
            epoch_number + 1
            )
        writer.add_scalars(
            'Metrics',
            {
                'Accuracy': acc.compute().item(),
                'Precision': prec.compute().item(),
                'Recall': rec.compute().item(),
                'F1 Score': f1.compute().item(),
                'Jaccard Index': jaccard.compute().item(),
            }
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            save_model(model, epoch_number, optimizer, avg_loss, save_dir, name)
        
def train_one_epoch(optimizer, loss_function, model, data_loader, batch_size, device, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Training epoch:'):
        inputs, labels, times = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        times = times.to(device) if isinstance(times, torch.Tensor) else times

        # Zero the gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs) if not isinstance(model, UTAE) else model(inputs, batch_positions=times)
        
        # Compute the loss and its gradients
        loss = loss_function(outputs, labels.long())
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            last_loss = running_loss / batch_size
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        # Garbage collection
        del inputs, labels, times, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return last_loss

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