import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .config import TIERS, get_device, layer_index


class Bundle:
    """One base model with all organism LoRA adapters loaded; adapters are toggled, never merged.

    `using(None)` disables every adapter (base model); `using(name)` activates exactly one organism.
    PEFT applies only the active adapter, so inactive organisms do not affect the forward pass.
    """

    def __init__(self, tier: str, adapters=None, dtype=None, device=None, extra_adapters=None):
        """extra_adapters: {name: local_path} of additional LoRA adapters (e.g. control organisms trained here)."""
        cfg = TIERS[tier]
        self.tier = tier
        self.device = device or get_device()
        self.dtype = dtype or cfg["dtype"]
        self.tok = AutoTokenizer.from_pretrained(cfg["base"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=self.dtype)
        base.eval()
        names = adapters or list(cfg["adapters"])
        self.model = PeftModel.from_pretrained(base, cfg["adapters"][names[0]], adapter_name=names[0],
                                               autocast_adapter_dtype=False)
        for n in names[1:]:
            self.model.load_adapter(cfg["adapters"][n], adapter_name=n, autocast_adapter_dtype=False)
        for n, path in (extra_adapters or {}).items():
            self.model.load_adapter(path, adapter_name=n, autocast_adapter_dtype=False)
            names.append(n)
        # keep adapter weights in the base dtype so the adapted forward pass has the same numerics as the base one
        for pname, p in self.model.named_parameters():
            if "lora_" in pname and p.dtype != self.dtype:
                p.data = p.data.to(self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.adapters = names
        self.n_layers = self.hf.config.num_hidden_layers
        self.d_model = self.hf.config.hidden_size
        self.main_layer = layer_index(self.n_layers)

    @property
    def hf(self):
        return self.model.base_model.model  # the underlying Qwen2ForCausalLM

    @property
    def layers(self):
        return self.hf.model.layers

    @contextlib.contextmanager
    def using(self, adapter):
        if adapter is None:
            with self.model.disable_adapter():
                yield
        else:
            self.model.set_adapter(adapter)
            yield

    def describe(self):
        return (f"tier={self.tier} base={TIERS[self.tier]['base']} layers={self.n_layers} d={self.d_model} "
                f"main_layer={self.main_layer} dtype={self.dtype} device={self.device} adapters={self.adapters}")
