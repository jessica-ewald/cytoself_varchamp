import inspect
import os
from os.path import join
from typing import Optional, Union, Collection, Sequence
from warnings import warn
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class BaseTrainer:
    """
    Base class for Trainer
    """

    def __init__(
        self,
        train_args: dict,
        metrics_names: Collection[str] = ('loss',),
        homepath: str = './',
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.train_args = train_args
        self.model = None
        self.best_model = []
        self.lr = 0
        self.tb_writer = None
        self.optimizer = None
        self.savepath_dict = {'homepath': homepath}
        self.current_epoch = 0
        self.metrics_names = metrics_names
        self.losses = {}

    def _init_model(self, model):
        """
        Initializes model

        Parameters
        ----------
        model : model object
            The autoencoder model
        Returns
        -------
        None

        """
        self.model = model
        self.model.to(self.device)
        # optimizer should be set after model moved to other devices
        self._default_train_args()
        self.set_optimizer(**self.train_args)
        self.init_savepath()

    def _default_train_args(self):
        """
        Sets default training arguments to make sure the model trainer has at least the following arguments.

        Returns
        -------
        None

        """
        args = {
            'reducelr_patience': 4,
            'reducelr_increment': 0.1,
            'earlystop_patience': 12,
            'min_lr': 1e-8,
            'max_epochs': 100,
        }
        for key, val in args.items():
            if key not in self.train_args:
                self.train_args[key] = val

    def calc_loss_one_batch(self, inputs, targets, **kwargs):
        """
        Computes loss for one batch

        Parameters
        ----------
        inputs : tensor
            input data
        targets : tensor
            target data
        kwargs : dict
            kwargs for the loss function

        Returns
        -------
        Tuple of tensors

        """
        return (nn.MSELoss(**kwargs)(targets, inputs),)

    def _adaptive_record_metrics(self, key: str, metrics: float):
        if key in self.losses:
            self.losses[key].append(metrics)
        else:
            self.losses[key] = [metrics]

    def record_metrics(self, metrics: Union[float, list], phase: str = 'train'):
        if not isinstance(metrics, list):
            metrics = [metrics]
        for n, l in zip(self.metrics_names, metrics):
            if isinstance(l, list):
                for i, _l in enumerate(l):
                    self._adaptive_record_metrics(f'{phase}_{n}{i + 1}', _l)
            else:
                self._adaptive_record_metrics(f'{phase}_{n}', l)

    def set_optimizer(self, optimizer: Optional = None, **kwargs):
        """
        Sets optimizer

        Parameters
        ----------
        optimizer : pytorch optimizer
            optimizer

        Returns
        -------
        None

        """
        if self.model:
            local_optimizer = optim.Adam if optimizer is None else optimizer
            local_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(local_optimizer).args if a in kwargs}
            self.optimizer = local_optimizer(self.model.parameters(), **local_kwargs)
        else:
            raise ValueError("self.model attribute is not initialized...")

    def enable_tensorboard(self, savepath='tb_logs', **kwargs):
        """
        Enables TensorBoard

        Parameters
        ----------
        savepath : str
            save path for tensorboard log

        Returns
        -------
        None

        """
        if 'tb_logs' in self.savepath_dict:
            warn(
                f'TensorBoard save path has been changed from {self.savepath_dict["tb_logs"]} to {savepath}',
                UserWarning,
            )
        self.savepath_dict['tb_logs'] = savepath
        if not os.path.exists(self.savepath_dict['tb_logs']):
            os.makedirs(self.savepath_dict['tb_logs'])
        self.tb_writer = SummaryWriter(self.savepath_dict['tb_logs'])

    def write_on_tensorboard(self, tensorboard_path: str, main_tag: str = 'Training'):
        if tensorboard_path is not None:
            if self.tb_writer is None:
                self.enable_tensorboard(tensorboard_path)
            self.tb_writer.add_scalars(
                main_tag,
                {key: val[-1] for key, val in self.losses.items() if 'test' not in key},
                self.current_epoch + 1,
            )
            self.tb_writer.flush()

    def init_savepath(self, makedirs: bool = True, **kwargs):
        """
        Initializes saving folders

        Parameters
        ----------
        makedirs : bool
            make directories if True

        Returns
        -------
        None

        """
        directories = ['checkpoints', 'embeddings', 'ft_analysis', 'umaps', 'visualization']
        for d in directories:
            self.savepath_dict[d] = join(self.savepath_dict['homepath'], d)
            if makedirs and not os.path.exists(self.savepath_dict[d]):
                os.makedirs(self.savepath_dict[d])

    def train_one_epoch(self, data_loader, **kwargs):
        """
        Trains self.model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.

        Returns
        -------
        None

        """
        if self.model is None:
            raise ValueError('model is not defined.')
        else:
            _metrics = [0] * len(self.metrics_names)
            for i, _batch in tqdm(enumerate(data_loader, desc='Train')):
                timg = self._get_data_by_name(_batch, 'image')
                self.optimizer.zero_grad()

                loss = self.calc_loss_one_batch(self.model(timg), timg, **kwargs)
                loss[0].backward()

                # Adjust learning weights
                self.optimizer.step()

                # Accumulate metrics
                _metrics = [m + l.item() for m, l in zip(_metrics, loss)]
            _metrics = [m / i for m in _metrics]
            self.record_metrics(_metrics, phase='train')

    @torch.inference_mode()
    def _infer_one_epoch(self, data_loader, _model):
        """
        Infers the output of a given model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.
        _model : model
            A model object
        Returns
        -------
        Numpy array

        """
        output, output_label = [], []
        for i, _batch in enumerate(data_loader, desc='Infer'):
            timg = self._get_data_by_name(_batch, 'image')
            out = _model(timg)
            if not torch.is_tensor(out):
                out = out[0]
            output.append(out.detach().cpu().numpy())
            if 'label' in _batch:
                output_label.append(_batch['label'])
        if len(output_label) == len(output):
            output_label = np.vstack(output_label)
        else:
            output_label = np.array([])
        return np.vstack(output), output_label

    def _get_data_by_name(self, data: dict, name: str):
        """
        Get tensor by name when the output of dataloader is dict.

        Parameters
        ----------
        data : dict
            Dictionary of tensor
        name : str
            Key of dict

        Returns
        -------

        """
        return data[name].float().to(self.device)

    @torch.inference_mode()
    def calc_val_loss(self, data_loader, **kwargs):
        """
        Compute validate loss

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader for validation data

        Returns
        -------
        Validation loss

        """
        if self.model is None:
            raise ValueError('model is not defined.')
        else:
            _metrics = [0] * len(self.metrics_names)
            for i, _batch in enumerate(tqdm(data_loader, desc='Val  ')):
                vimg = self._get_data_by_name(_batch, 'image')
                _vloss = self.calc_loss_one_batch(self.model(vimg), vimg)
                _metrics = [m + l.item() for m, l in zip(_metrics, _vloss)]
            self.record_metrics(_metrics, phase='val')

    def _reduce_lr_on_plateau(self, count_lr_no_improve: int):
        """
        Reduces learning rate when no improvement in the training

        Parameters
        ----------
        count_lr_no_improve : int
            Number of epochs with no improvement

        Returns
        -------
        int

        """
        if self.optimizer is None:
            raise ValueError('optimizer is not defined.')
        else:
            if count_lr_no_improve >= self.train_args['reducelr_patience']:
                if self.optimizer.param_groups[0]['lr'] > self.train_args['min_lr']:
                    self.optimizer.param_groups[0]['lr'] *= self.train_args['reducelr_increment']
                    print('learn rate = ', self.optimizer.param_groups[0]['lr'])
                    return 0
                else:
                    return count_lr_no_improve
            else:
                return count_lr_no_improve

    def _detach_graph(self, losses: Union[Sequence[Tensor], Tensor]):
        """
        Detach graph from loss object to save memory

        Parameters
        ----------
        losses : list or loss object

        Returns
        -------
        Tensor or list of Tensor

        """
        if isinstance(losses, Tensor):
            return losses.detach()
        else:
            return [loss.detach() for loss in losses]

    def fit(
        self,
        datamanager,
        initial_epoch: int = 0,
        tensorboard_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Fit pytorch model

        Parameters
        ----------
        datamanager : DataManager
            DataManager object
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous training run).
        tensorboard_path : str
            Path for Tensorboard to load logs

        Returns
        -------
        None

        """
        if self.model is None:
            raise ValueError('model is not defined.')
        else:
            self.current_epoch = initial_epoch
            best_vloss = torch.inf if 'val_loss' not in self.losses else min(self.losses['val_loss'])
            count_lr_no_improve = 0
            count_early_stop = 0
            for current_epoch in range(self.current_epoch, self.train_args['max_epochs']):
                print(f'Epoch {current_epoch}/{self.train_args["max_epochs"]}')
                # Train the model
                self.model.train(True)
                self.train_one_epoch(datamanager.train_loader, **kwargs)
                self.model.train(False)

                # Validate the model
                self.calc_val_loss(datamanager.val_loader)
                _vloss = self.losses['val_loss'][-1]

                # Track the best performance, and save the model's state
                if _vloss < best_vloss:
                    best_vloss = _vloss
                    self.best_model = self.model.state_dict()
                else:
                    count_lr_no_improve += 1
                    count_early_stop += 1

                # Reduce learn rate on plateau
                count_lr_no_improve = self._reduce_lr_on_plateau(count_lr_no_improve)

                # Record logs for TensorBoard
                if tensorboard_path is not None:
                    tensorboard_path = join(self.savepath_dict['homepath'], tensorboard_path)
                self.write_on_tensorboard(tensorboard_path)
                self.current_epoch += 1

                # Check for early stopping
                if count_early_stop >= self.train_args['earlystop_patience']:
                    print('Early stopping.')
                    break

            torch.save(self.best_model, join(self.savepath_dict['homepath'], f'model_{self.current_epoch + 1}.pt'))

    @torch.inference_mode()
    def infer_embeddings(self, data):
        """
        Infers embeddings

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data

        Returns
        -------
        None

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self._infer_one_epoch(data, self.model.encoder)
        else:
            return self.model.encoder(torch.from_numpy(data).float().to(self.device)).detach().cpu().numpy()

    @torch.inference_mode()
    def infer_reconstruction(self, data):
        """
        Infers decoded images

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data

        Returns
        -------
        None

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self._infer_one_epoch(data, self.model)[0]
        else:
            output = self.model(torch.from_numpy(data).float().to(self.device))
            if isinstance(output, tuple) or isinstance(output, list):
                return output[0].detach().cpu().numpy()
            else:
                return output.detach().cpu().numpy()
