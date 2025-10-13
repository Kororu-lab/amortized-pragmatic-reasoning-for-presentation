"""
Configuration management for the amortized RSA project.
Centralizes paths, hyperparameters, and model configurations.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import constants


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Root directories
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_root: Path = field(default_factory=lambda: Path(__file__).parent / 'data')
    models_root: Path = field(default_factory=lambda: Path(__file__).parent / 'models')
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.project_root = Path(self.project_root)
        self.data_root = Path(self.data_root)
        self.models_root = Path(self.models_root)
    
    def get_dataset_dir(self, dataset: str) -> Path:
        """Get directory for a specific dataset."""
        return self.data_root / dataset
    
    def get_model_dir(self, dataset: str, generalization: Optional[str] = None) -> Path:
        """Get directory for model files."""
        if generalization:
            return self.models_root / dataset / 'generalization' / generalization
        return self.models_root / dataset
    
    def get_literal_listener_path(self, dataset: str, index: int = 0, 
                                   generalization: Optional[str] = None) -> Path:
        """Get path to a literal listener model."""
        model_dir = self.get_model_dir(dataset, generalization)
        return model_dir / f'literal_listener_{index}.pt'
    
    def get_literal_speaker_path(self, dataset: str, 
                                  generalization: Optional[str] = None) -> Path:
        """Get path to a literal speaker model."""
        model_dir = self.get_model_dir(dataset, generalization)
        return model_dir / 'literal_speaker.pt'
    
    def get_amortized_speaker_path(self, dataset: str,
                                    generalization: Optional[str] = None) -> Path:
        """Get path to an amortized speaker model."""
        model_dir = self.get_model_dir(dataset, generalization)
        return model_dir / 'amortized_speaker.pt'
    
    def get_reinforce_speaker_path(self, dataset: str,
                                    generalization: Optional[str] = None) -> Path:
        """Get path to a REINFORCE speaker model."""
        model_dir = self.get_model_dir(dataset, generalization)
        return model_dir / 'reinforce_speaker.pt'
    
    def get_language_model_path(self, dataset: str) -> Path:
        """Get path to a language model."""
        return self.models_root / dataset / 'language_model.pt'
    
    def get_vocab_path(self, dataset: str) -> Path:
        """Get path to vocabulary file."""
        return self.models_root / dataset / 'vocab.pt'


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    epochs: int = constants.DEFAULT_EPOCHS
    learning_rate: Optional[float] = None
    
    # Model-specific parameters
    model_type: str = 'amortized'  # s0, l0, amortized, sample, rsa
    activation: Optional[str] = None  # gumbel, softmax, multinomial
    penalty: Optional[str] = None  # length
    lmbd: float = constants.DEFAULT_LAMBDA
    tau: float = constants.DEFAULT_TAU
    
    # Training settings
    cuda: bool = False
    debug: bool = False
    
    # Data settings
    dataset: str = 'shapeworld'
    generalization: Optional[str] = None  # new_color, new_combo, new_shape
    
    def get_learning_rate(self) -> float:
        """Get learning rate based on model type if not explicitly set."""
        if self.learning_rate is not None:
            return self.learning_rate
        
        if self.model_type == 'l0':
            return constants.DEFAULT_LEARNING_RATE_L0
        return constants.DEFAULT_LEARNING_RATE_S0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_models = {'s0', 'l0', 'amortized', 'sample', 'rsa', 'oracle', 'test', 'language_model'}
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_models}")
        
        valid_activations = {None, 'gumbel', 'softmax', 'softmax_noise', 'multinomial'}
        if self.activation not in valid_activations:
            raise ValueError(f"Invalid activation: {self.activation}. Must be one of {valid_activations}")
        
        valid_penalties = {None, 'length'}
        if self.penalty not in valid_penalties:
            raise ValueError(f"Invalid penalty: {self.penalty}. Must be one of {valid_penalties}")
        
        valid_datasets = {'shapeworld', 'colors'}
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.dataset}. Must be one of {valid_datasets}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Embedding parameters
    embedding_dim: int = constants.DEFAULT_EMBEDDING_DIM
    hidden_size: int = constants.DEFAULT_HIDDEN_SIZE
    
    # Vision model
    vision_model: str = 'Conv4'  # Conv4, Conv6, ResNet18, etc.
    
    # Sequence parameters
    max_sequence_length: int = constants.MAX_SEQUENCE_LENGTH


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    dataset: str = 'shapeworld'
    generalization: Optional[str] = None
    
    # ShapeWorld data files
    def get_shapeworld_pretrain_files(self, generalization: Optional[str] = None) -> List[List[str]]:
        """Get pretrain data file paths for ShapeWorld."""
        if generalization:
            data_dir = f'./data/shapeworld/generalization/{generalization}/reference-1000-'
            return [
                [f'{data_dir}{i}.npz' for i in range(5)],
                [f'{data_dir}{i}.npz' for i in range(5, 10)]
            ]
        else:
            data_dir = './data/shapeworld/reference-1000-'
            files = []
            for start in range(0, 75, 5):
                if start == 55:  # Skip 55-69
                    continue
                if start == 70:  # Only include 70-74
                    files.append([f'{data_dir}{i}.npz' for i in range(70, 75)])
                else:
                    files.append([f'{data_dir}{i}.npz' for i in range(start, start + 5)])
            return files
    
    def get_shapeworld_train_files(self, generalization: Optional[str] = None) -> List[str]:
        """Get training data file paths for ShapeWorld."""
        if generalization:
            data_dir = f'./data/shapeworld/generalization/{generalization}/reference-1000-'
        else:
            data_dir = './data/shapeworld/reference-1000-'
        return [f'{data_dir}{i}.npz' for i in range(60, 65)]
    
    def get_shapeworld_val_files(self, generalization: Optional[str] = None) -> List[str]:
        """Get validation data file paths for ShapeWorld."""
        if generalization:
            data_dir = f'./data/shapeworld/generalization/{generalization}/reference-1000-'
        else:
            data_dir = './data/shapeworld/reference-1000-'
        return [f'{data_dir}{i}.npz' for i in range(65, 70)]
    
    def get_colors_pretrain_files(self) -> List[List[str]]:
        """Get pretrain data file paths for Colors dataset."""
        data_dir = './data/colors/data_1000_'
        return [
            [f'{data_dir}{i}.npz' for i in range(15)],
            [f'{data_dir}{i}.npz' for i in range(15, 30)],
            [f'{data_dir}{i}.npz' for i in range(30, 45)]
        ]
    
    def get_colors_train_files(self) -> List[str]:
        """Get training data file paths for Colors dataset."""
        data_dir = './data/colors/data_1000_'
        return [f'{data_dir}{i}.npz' for i in range(15)]
    
    def get_colors_val_files(self) -> List[str]:
        """Get validation data file paths for Colors dataset."""
        data_dir = './data/colors/data_1000_'
        return [f'{data_dir}{i}.npz' for i in range(15, 30)]


@dataclass
class Config:
    """Main configuration object combining all sub-configurations."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.training.validate()
    
    @classmethod
    def from_args(cls, args) -> 'Config':
        """Create configuration from argparse arguments."""
        config = cls()
        
        # Training config
        config.training.batch_size = args.batch_size
        config.training.epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.cuda = args.cuda
        config.training.debug = args.debug
        config.training.dataset = args.dataset
        
        # Model-specific args
        if hasattr(args, 's0') and args.s0:
            config.training.model_type = 's0'
        elif hasattr(args, 'l0') and args.l0:
            config.training.model_type = 'l0'
        elif hasattr(args, 'amortized') and args.amortized:
            config.training.model_type = 'amortized'
        
        if hasattr(args, 'activation'):
            config.training.activation = args.activation
        if hasattr(args, 'penalty'):
            config.training.penalty = args.penalty
        if hasattr(args, 'lmbd'):
            config.training.lmbd = args.lmbd
        if hasattr(args, 'tau'):
            config.training.tau = args.tau
        if hasattr(args, 'generalization'):
            config.training.generalization = args.generalization
            config.data.generalization = args.generalization
        
        config.validate()
        return config

