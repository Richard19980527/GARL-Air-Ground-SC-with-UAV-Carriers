from util import *
from methods.net_tools import *


FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


class MultiCategorical(torch.distributions.Distribution):
    def __init__(self, dists: List[torch.distributions.Categorical]):
        super().__init__(validate_args=False)
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1, keepdim=True)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)


class MultiCategorical_Generator(nn.Module):
    def __init__(self, input_size, choice_num_list: List[int]):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.choice_num_list = choice_num_list
        output_size = sum(self.choice_num_list)
        self.linear = nn.Sequential(
            init_(nn.Linear(input_size, output_size)),
            nn.Sigmoid(),
        )

    def forward(self, x, x_mask):
        x = self.linear(x)
        x = torch.mul(x, x_mask)
        start = 0
        ans = []
        for n in self.choice_num_list:
            ans.append(torch.distributions.Categorical(probs=x[:, start:start + n]))
            start += n
        return MultiCategorical(ans)


class DiagGaussian(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))
        self.fc_mean = nn.Sequential(
            init_(nn.Linear(input_size, output_size)),
        )
        self.logstd = AddBias(torch.zeros(output_size, dtype=torch.float32))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)
        zeros = torch.zeros(action_mean.size(), device=x.device)
        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)
        return FixedNormal(action_mean, action_logstd.exp())
