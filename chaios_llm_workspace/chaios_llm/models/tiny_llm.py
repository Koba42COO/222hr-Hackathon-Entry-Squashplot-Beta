"""
Tiny LLM implementation for testing and development
Based on the freeCodeCamp guide: "Code an LLM From Scratch"
"""

import torch
import torch.nn as nn
from ..src.transformer import chAIosLLM
from ..src.config import TinyLLMConfig


class TinyLLM(chAIosLLM):
    """A tiny version of chAIos LLM for quick experimentation and testing."""

    def __init__(self, config: TinyLLMConfig):
        super().__init__(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_path: str) -> "TinyLLM":
        """Load a pretrained TinyLLM model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = TinyLLMConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_pretrained(self, save_path: str):
        """Save the model and its configuration."""
        torch.save({
            'config': self.config.__dict__,
            'model_state_dict': self.state_dict(),
        }, save_path)

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get the model size in megabytes."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


def create_tiny_llm(vocab_size: int = 10000) -> TinyLLM:
    """Factory function to create a TinyLLM with default configuration."""
    config = TinyLLMConfig(vocab_size=vocab_size)
    return TinyLLM(config)


def create_tiny_llm_from_config(config_dict: dict) -> TinyLLM:
    """Create a TinyLLM from a configuration dictionary."""
    config = TinyLLMConfig(**config_dict)
    return TinyLLM(config)


# Example usage and testing functions
def test_tiny_llm():
    """Test function to verify TinyLLM works correctly."""
    model = create_tiny_llm()
    print(f"TinyLLM created with {model.count_parameters():,} parameters")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")

    # Test forward pass
    batch_size, seq_length = 2, 50
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))

    with torch.no_grad():
        logits, loss = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss}")

    # Test generation
    generated = model.generate(input_ids[:, :10], max_new_tokens=20)
    print(f"Generated sequence shape: {generated.shape}")

    return model


if __name__ == "__main__":
    test_tiny_llm()
