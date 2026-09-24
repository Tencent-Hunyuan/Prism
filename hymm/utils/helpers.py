
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Dict

@dataclass
class BaseStates:
    epoch: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset_epoch_based_states(self):
        pass

    def inc_epoch(self):
        """ Increase (image) epoch by 1. """
        self.epoch += 1
        self.reset_epoch_based_states()
        return self.epoch

    def __repr__(self):
        max_key_length = max([len(k) for k in asdict(self).keys()])
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def to_dict(self):
        states = deepcopy(vars(self))
        # Convert defaultdict to dict
        for key, value in states.items():
            if isinstance(value, defaultdict):
                states[key] = dict(value)

        return states

    @classmethod
    def from_pretrained(cls, scalar_state, rank=None, world_size=None, default_rank0_ss=None, default=None):
        _scalar_state = default or {}
        if isinstance(scalar_state, dict):
            _scalar_state.update(scalar_state)
        elif isinstance(scalar_state, list):
            assert rank is not None and world_size is not None and default_rank0_ss is not None, (
                f"rank({rank}), world_size({world_size}), and default_rank0_ss({default_rank0_ss}) should be provided "
                f"when scalar_state is a list."
            )
            if len(scalar_state) != world_size:
                # Check if scalar_state all the same
                if all([scalar_state[0] == x for x in scalar_state]):
                    default_rank0_ss = True

                if default_rank0_ss:
                    from loguru import logger
                    logger.info(f" len(scalar_state)={len(scalar_state)} != world_size={world_size}, Set default rank0 scalar state.")
                    _scalar_state.update(scalar_state[0])
                else:
                    _scalar_state.update(scalar_state[0])
                    for i in range(len(scalar_state)):
                        if isinstance(scalar_state[i], dict):
                            for k, v in scalar_state[i].items():
                                if isinstance(v, dict):
                                    _scalar_state[k].update(v)
                    from loguru import logger
                    logger.info(f"default_rank0_ss is alse, Update scalar_state={_scalar_state}")
            else:
                _scalar_state.update(scalar_state[rank])
        else:
            raise ValueError(f"Unknown scalar state type: {type(scalar_state)}")
        
        # Convert all dict fields back to defaultdict(int).
        # to_dict() converts defaultdict → dict for pickle serialization;
        # we must reverse this for every field that was originally defaultdict,
        # otherwise accessing a new key (e.g., after config change) raises KeyError.
        _defaultdict_int_fields = [
            'epoch',
            'consumed_samples_total',
            'epoch_consumed_samples',
            'consumed_samples_per_dp',
            'consumed_tokens_total',
            'consumed_samples_by_mask_type_total',
            'consumed_samples_by_mask_type_per_dp',
            'consumed_samples_by_data_type_total',
            'consumed_samples_by_data_type_per_dp',
        ]
        for field_name in _defaultdict_int_fields:
            if field_name in _scalar_state and isinstance(_scalar_state[field_name], dict):
                _scalar_state[field_name] = defaultdict(int, _scalar_state[field_name])
        
        return cls(**_scalar_state)


@dataclass
class ScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """
    # - Global level
    epoch: dict = field(default_factory=lambda: defaultdict(int))
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = 0  # Update steps in current run (Reset for every resume)
    consumed_samples_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples
    epoch_consumed_samples: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples

    consumed_samples_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples per data-parallel group
    consumed_tokens_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed tokens
    consumed_computations_attn: float = 0         # Accumulated consumed computations of attention + mlp
    consumed_computations_total: float = 0        # Accumulated consumed computations of total
    
    # - Mask type statistics
    consumed_samples_by_mask_type_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by mask type
    consumed_samples_by_mask_type_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by mask type per data-parallel group
    
    # - Data type statistics for t2v mask type
    consumed_samples_by_data_type_total: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by data type for t2v
    consumed_samples_by_data_type_per_dp: dict = field(default_factory=lambda: defaultdict(int))  # Accumulated consumed samples by data type per data-parallel group for t2v
    
    lr: float = 0.0  # Current learning rate


@dataclass
class CycleStates(BaseStates):
    log_steps: int = 0
    running_loss: float = 0
    running_tokens: dict = field(default_factory=lambda: defaultdict(int))
    running_samples: dict = field(default_factory=lambda: defaultdict(int))
    running_samples_by_mask_type: dict = field(default_factory=lambda: defaultdict(int))
    running_loss_by_mask_type: dict = field(default_factory=lambda: defaultdict(float))
    running_loss_count_by_mask_type: dict = field(default_factory=lambda: defaultdict(int))
    # Add data_type tracking for t2v mask type
    running_loss_by_data_type: dict = field(default_factory=lambda: defaultdict(float))
    running_loss_count_by_data_type: dict = field(default_factory=lambda: defaultdict(int))
    running_samples_by_data_type: dict = field(default_factory=lambda: defaultdict(int))
    
    def reset_epoch_based_states(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_tokens = defaultdict(int)
        self.running_samples = defaultdict(int)
        self.running_samples_by_mask_type = defaultdict(int)
        self.running_loss_by_mask_type = defaultdict(float)
        self.running_loss_count_by_mask_type = defaultdict(int)
        self.running_loss_by_data_type = defaultdict(float)
        self.running_loss_count_by_data_type = defaultdict(int)
        self.running_samples_by_data_type = defaultdict(int)
    
    def add_mask_type_samples(self, mask_type, samples):
        self.running_samples_by_mask_type[mask_type] += samples
    
    def add_mask_type_loss(self, mask_type, loss_value):
        self.running_loss_by_mask_type[mask_type] += loss_value
        self.running_loss_count_by_mask_type[mask_type] += 1
    
    def add_data_type_loss(self, data_type, loss_value, samples):
        """Add loss and samples for specific data_type"""
        self.running_loss_by_data_type[data_type] += loss_value
        self.running_loss_count_by_data_type[data_type] += 1
        self.running_samples_by_data_type[data_type] += samples
    
    def get_avg_loss_by_mask_type(self):
        """获取各mask_type的平均loss"""
        avg_losses = {}
        for mask_type in self.running_loss_by_mask_type.keys():
            count = self.running_loss_count_by_mask_type[mask_type]
            if count > 0:
                avg_losses[mask_type] = self.running_loss_by_mask_type[mask_type] / count
            else:
                avg_losses[mask_type] = 0.0
        return avg_losses
    
    def get_avg_loss_by_data_type(self):
        """获取各data_type的平均loss"""
        avg_losses = {}
        for data_type in self.running_loss_by_data_type.keys():
            count = self.running_loss_count_by_data_type[data_type]
            if count > 0:
                avg_losses[data_type] = self.running_loss_by_data_type[data_type] / count
            else:
                avg_losses[data_type] = 0.0
        return avg_losses
