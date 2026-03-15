import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossPatchSublayer3D(nn.Module):
    def __init__(self, mlp_ratio=4.0):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.norm = None
        self.mlp = None
        self.num_tokens = None

    def _build(self, num_tokens, device, dtype):
        hidden_tokens = max(1, int(num_tokens * self.mlp_ratio))
        self.norm = nn.LayerNorm(num_tokens).to(device=device, dtype=dtype)
        self.mlp = Mlp(num_tokens, hidden_tokens, num_tokens).to(device=device, dtype=dtype)
        self.num_tokens = num_tokens

    def forward(self, x):
        b, c, d, h, w = x.shape
        n = d * h * w
        if (self.norm is None) or (self.num_tokens != n):
            self._build(n, x.device, x.dtype)
        residual = x
        x = x.reshape(b, c, n)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.reshape(b, c, d, h, w)
        x = x + residual
        return x


class CrossChannelSublayer3D(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = max(1, int(channels * mlp_ratio))
        self.norm = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, hidden_dim, channels)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x + residual
        return x


class MLPLike3D(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.cross_patch = CrossPatchSublayer3D(mlp_ratio=mlp_ratio)
        self.cross_channel = CrossChannelSublayer3D(channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = self.cross_patch(x)
        x = self.cross_channel(x)
        return x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class HighFrequencyBranch3D(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if in_channels < 2:
            raise ValueError("High-frequency branch needs at least 2 channels.")
        self.c1 = in_channels // 2
        self.c2 = in_channels - self.c1
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_after_pool = nn.Conv3d(self.c1, self.c1, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Conv3d(self.c2, self.c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.dsconv = DepthwiseSeparableConv3d(self.c2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x1 = x[:, :self.c1, :, :, :].contiguous()
        x2 = x[:, self.c1:, :, :, :].contiguous()
        y1 = self.conv_after_pool(self.maxpool(x1))
        y2 = self.dsconv(self.fc(x2))
        y = torch.cat([y1, y2], dim=1)
        return y


class LowFrequencyBranch3D(nn.Module):
    def __init__(self, in_channels, mlp_ratio=4.0, pool_kernel=2, pool_stride=2):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride, ceil_mode=False)
        self.mlp_like = MLPLike3D(in_channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        target_size = x.shape[2:]
        x = self.pool(x)
        x = self.mlp_like(x)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return x


class IMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        high_freq_ratio=0.5,
        mlp_ratio=4.0,
        low_pool_kernel=2,
        low_pool_stride=2,
        kernel_size=3
    ):
        super().__init__()

        high_dim_float = in_channels * high_freq_ratio
        if abs(high_dim_float - round(high_dim_float)) > 1e-6:
            raise ValueError(
                f"high_freq_ratio={high_freq_ratio} makes high_dim non-integer for in_channels={in_channels}. Got {high_dim_float}."
            )

        self.high_dim = int(round(high_dim_float))
        self.low_dim = in_channels - self.high_dim

        if self.high_dim <= 0 or self.low_dim <= 0:
            raise ValueError(
                f"Invalid split: high_dim={self.high_dim}, low_dim={self.low_dim}."
            )

        padding = kernel_size // 2

        self.high_branch = HighFrequencyBranch3D(
            in_channels=self.high_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        self.low_branch = LowFrequencyBranch3D(
            in_channels=self.low_dim,
            mlp_ratio=mlp_ratio,
            pool_kernel=low_pool_kernel,
            pool_stride=low_pool_stride
        )

        fusion_in_channels = self.high_dim + self.low_dim
        self.fuse_conv = nn.Conv3d(
            fusion_in_channels,
            fusion_in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.fuse_bn = nn.BatchNorm3d(fusion_in_channels)
        self.fuse_relu = nn.ReLU(inplace=True)

        self.proj = nn.Conv3d(
            fusion_in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        xh = x[:, :self.high_dim, :, :, :].contiguous()
        xl = x[:, self.high_dim:, :, :, :].contiguous()
        yh = self.high_branch(xh)
        yl = self.low_branch(xl)
        yc = torch.cat([yh, yl], dim=1)
        y = self.fuse_relu(yc + self.fuse_bn(self.fuse_conv(yc)))
        y = self.proj(y)
        return y


class iMlpLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        bias=True,
        normalization=True,
        high_freq_ratio=0.5,
        mlp_ratio=4.0,
        low_pool_kernel=2,
        low_pool_stride=2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.normalization = normalization
        self.kernel_size = kernel_size
        self.bias = bias

        ks = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size

        self.imlp = IMLP(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            high_freq_ratio=high_freq_ratio,
            mlp_ratio=mlp_ratio,
            low_pool_kernel=low_pool_kernel,
            low_pool_stride=low_pool_stride,
            kernel_size=ks
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.imlp(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        d, h, w = image_size
        weight = self.imlp.proj.weight
        h0 = torch.zeros(batch_size, self.hidden_dim, d, h, w, device=weight.device, dtype=weight.dtype)
        c0 = torch.zeros(batch_size, self.hidden_dim, d, h, w, device=weight.device, dtype=weight.dtype)
        return h0, c0


class iMlpLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        high_freq_ratio=0.5,
        batch_first=True,
        bias=True,
        normalization=True,
        return_all_layers=False,
        mlp_ratio=4.0,
        low_pool_kernel=2,
        low_pool_stride=2
    ):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not (len(kernel_size) == len(hidden_dim) == num_layers):
            raise ValueError("Inconsistent list length among kernel_size / hidden_dim.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.normalization = normalization

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell = iMlpLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim[i],
                kernel_size=kernel_size[i],
                bias=bias,
                normalization=normalization,
                high_freq_ratio=high_freq_ratio,
                mlp_ratio=mlp_ratio,
                low_pool_kernel=low_pool_kernel,
                low_pool_stride=low_pool_stride
            )
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5).contiguous()

        b, seq_len, _, d, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError("Passing hidden_state is not implemented in this version.")
        hidden_state = self._init_hidden(batch_size=b, image_size=(d, h, w))

        cur_layer_input = input_tensor
        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h_t, c_t = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h_t, c_t = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :, :],
                    cur_state=(h_t, c_t)
                )
                output_inner.append(h_t)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append((h_t, c_t))

        if self.return_all_layers:
            return layer_output_list, last_state_list
        return layer_output_list[-1], last_state_list[-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple) or
            (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
