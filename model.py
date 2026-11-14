import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer


_TEXT_ENCODER_CACHE = {}


def load_text_encoder(model_path, device, local_files_only=False, num_layers=2):
    key = (model_path, str(device), bool(local_files_only), int(num_layers))
    if key in _TEXT_ENCODER_CACHE:
        return _TEXT_ENCODER_CACHE[key]

    config = GPT2Config.from_pretrained(
        model_path, local_files_only=local_files_only)
    config.num_hidden_layers = num_layers
    config.output_attentions = False
    config.output_hidden_states = False

    model = GPT2Model.from_pretrained(
        model_path, config=config, local_files_only=local_files_only
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    _TEXT_ENCODER_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def default_description(class_names):
    names = ", ".join(class_names)
    return (f"The dataset contains {len(class_names)} underwater acoustic "
            f"classes: {names}. Task: classification.")


class SpectralAdapter(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_ff),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1.0 - alpha) * self.ffn2(self.ffn1(x))


class BidirectionalMamba(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=4):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state,
                                   d_conv=d_conv, expand=expand)
        self.reverse_mamba = Mamba(d_model=d_model, d_state=d_state,
                                   d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        y_fwd = self.forward_mamba(x)
        y_rev = self.reverse_mamba(x.flip(1)).flip(1)
        return self.proj(torch.cat([y_fwd, y_rev], dim=-1))


class PaperKMA(nn.Module):
    def __init__(
        self,
        classes,
        d_model=128,
        d_ff=512,
        llm_dim=768,
        dropout=0.1,
        description=None,
        gpt2_path="gpt2",
        local_files_only=False,
        text_layers=2,
    ):
        super().__init__()
        self.num_classes = classes
        self.d_model = d_model
        self.description = description
        self.gpt2_path = gpt2_path
        self.local_files_only = local_files_only
        self.text_layers = text_layers

        # ===== Spectral =====
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),
        )
        self.adapter = SpectralAdapter(d_model, d_ff, dropout)

        # ===== Mamba =====
        self.bimamba = BidirectionalMamba(d_model)

        # ===== Text =====
        self.llm_out = nn.Linear(llm_dim, d_model)
        self.text_ln = nn.LayerNorm(d_model)

        # ===== Alignment =====
        self.align_v = nn.Linear(d_model, d_model)
        self.sim_q = nn.Linear(d_model, d_model, bias=False)
        self.sim_k = nn.Linear(d_model, d_model, bias=False)
        self.align_dropout = nn.Dropout(dropout)
        self.align_ln = nn.LayerNorm(d_model)

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes),
        )

    def _preprocess(self, x):
        x = self.spectral_conv(x)
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return self.adapter(x)

    def _text_prior(self, device, batch_size):
        if self.description is None:
            raise ValueError("description is not set")
        model, tokenizer = load_text_encoder(
            self.gpt2_path, device,
            local_files_only=self.local_files_only,
            num_layers=self.text_layers,
        )
        ids = tokenizer(
            self.description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).input_ids.to(device)

        with torch.no_grad():
            hidden = model(input_ids=ids).last_hidden_state
        prior = self.llm_out(hidden).mean(dim=1)
        return self.text_ln(prior).expand(batch_size, -1)

    def forward(self, x_main):
        z = self._preprocess(x_main)
        e_audio = self.bimamba(z)

        h_text = self._text_prior(e_audio.device, e_audio.size(0))
        q = F.normalize(self.sim_q(e_audio), dim=-1, eps=1e-6)
        k = F.normalize(self.sim_k(h_text), dim=-1, eps=1e-6)
        m_sim = torch.sigmoid((q * k.unsqueeze(1)).sum(dim=-1, keepdim=True))

        e_align = self.align_ln(
            self.align_dropout(self.align_v(e_audio) * m_sim) + e_audio)

        return self.classifier(e_align.permute(0, 2, 1))
