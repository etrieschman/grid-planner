
from tqdm import tqdm
import torch
import copy

def train_model(
    model, criterion, optimizer, scheduler, 
    dataloaders, dataset_sizes, path_model, device, epochs=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e20    
    # loop epochs
    loss_history = {'train':[], 'validate':[],'epoch_train':[], 'epoch_validate':[]}
    for epoch in range(epochs):
        # run training and validating modes
        for phase in ['train', 'validate']:
            model.train() if (phase == 'train') else model.eval()

            running_num, running_loss = 0.0, 0.0
            data_bar = tqdm(dataloaders[phase])
            # loop batches
            for x in data_bar:
                x = x.to(device)
                xhat, mu, logvar = model(x)
                loss = criterion(xhat, x, mu, logvar)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # estimate and print summary stats
                running_num += x.size(0)
                running_loss += loss.item()
                loss_history[phase].append((running_loss / running_num))
                data_bar.set_description(
                    '{} epoch: [{}/{}] Loss: {:.4f}'
                    .format(phase, epoch+1, epochs, running_loss / running_num))

            # update line search decay
            if (phase == 'train') & (scheduler is not None):
                scheduler.step()

            # tracking for best model
            epoch_loss = running_loss / dataset_sizes[phase]
            loss_history[f'epoch_{phase}'].append(epoch_loss)
            if phase == 'validate' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path_model / 'temp'/ 'best_current_model.pt')

    # load best model
    print(f'\nReturning best model, with validation loss {best_loss}')
    model.load_state_dict(best_model_wts)
    return best_model_wts, loss_history