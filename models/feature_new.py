import torch, torchaudio
import torch.nn.functional as F

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        #input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class OnlineFbank(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6 # eps
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
        return x
            
