"""
DreamerV2
Paper link: xxxx.pdf
Implementation: Pytorch
"""
import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class DreamerV2_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(DreamerV2_Learner, self).__init__(config, policy)
        self.optimizer = {
            'model': torch.optim.Adam(self.policy.model_parameters, self.config.learning_rate_model),
            'actor': torch.optim.Adam(self.policy.actor_parameters, self.config.learning_rate_actor),
            'critic': torch.optim.Adam(self.policy.critic_parameters, self.config.learning_rate_critic)}
        # self.scheduler = {
        #     'model': torch.optim.lr_scheduler.LinearLR(self.optimizer['model'],
        #                                                start_factor=1.0,
        #                                                end_factor=self.end_factor_lr_decay,
        #                                                total_iters=self.config.running_steps),
        #     'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
        #                                                start_factor=1.0,
        #                                                end_factor=self.end_factor_lr_decay,
        #                                                total_iters=self.config.running_steps),
        #     'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
        #                                                 start_factor=1.0,
        #                                                 end_factor=self.end_factor_lr_decay,
        #                                                 total_iters=self.config.running_steps)}
        # self.tau = config.tau
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        """(n_envs, seq, batch, ~)"""
        obs_seq_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_seq_batch = torch.as_tensor(samples['act'], device=self.device)
        # if not hasattr(self.config, 'action') or not self.config.action == "one-hot":
        # process to one_hot_act
        act_seq_batch = torch.nn.functional.one_hot(act_seq_batch.long(), num_classes=self.n_actions).float()
        rew_seq_batch = torch.as_tensor(samples['rew'], device=self.device)
        noterm_seq_batch = torch.as_tensor(samples['noterm'], device=self.device)

        """n_envs stack to batch"""
        """(n_envs, seq, batch, ~) -> (seq, n_envs, batch, ~) -> (seq, n_envs * batch, ~)"""
        n_envs, seq, batch = obs_seq_batch.shape[0:3]
        obs_shape = obs_seq_batch.shape[3:]
        obs_seq_batch = obs_seq_batch.transpose(0, 1).reshape(seq, n_envs * batch, *obs_shape)
        act_seq_batch = act_seq_batch.transpose(0, 1).reshape(seq, n_envs * batch, self.n_actions)
        rew_seq_batch = rew_seq_batch.transpose(0, 1).reshape(seq, n_envs * batch, 1)
        noterm_seq_batch = noterm_seq_batch.transpose(0, 1).reshape(seq, n_envs * batch, 1)

        """
        model learning
        model_loss = -obs_dist.log_prob(obs_batch) - rew_dist.log_prob(rew_batch) - noterm_dist.log_prob(noterm_batch) + 
                kl_div(prior, sg(post)) + kl_div(sg(prior), post)
        """
        obs_dist, rew_dist, noterm_dist, prior, post = self.policy.model_forward(
            obs_seq_batch,
            act_seq_batch,
            noterm_seq_batch
        )
        """(seq, n_envs * batch, ~)"""
        kl_div = torch.nn.functional.kl_div
        obs_loss = -torch.mean(obs_dist.log_prob(obs_seq_batch))
        rew_loss = -torch.mean(rew_dist.log_prob(rew_seq_batch))
        noterm_loss = -torch.mean(noterm_dist.log_prob(noterm_seq_batch))
        alpha = self.config.kl['kl_balance_scale']
        eps = 1e-8
        kl_loss = (alpha * kl_div(prior.stoch + eps, post.stoch.detach() + eps) +
                   (1 - alpha) * kl_div(prior.stoch.detach() + eps, post.stoch + eps))
        rew_scale = self.config.loss_scale['reward']
        discount_scale = self.config.loss_scale['discount']
        kl_scale = self.config.loss_scale['kl']
        model_loss = obs_loss + rew_scale * rew_loss + discount_scale * noterm_loss + kl_scale * kl_loss
        self.optimizer['model'].zero_grad()
        model_loss.backward()
        self.optimizer['model'].step()
        """
        actor_critic learning
        actor_loss = -rho * act_dist.log_prob(act_batch) * sg(V_lambda - target_v_dist.sample()) -
                (1 - rho) * V_lambda + act_dist.entropy()
        critic_loss = -v_dist.log_prob(sg(V_lambda))
        """
        act_log_probs, act_ent, imag_value, V_lambda, value_dist = self.policy.actor_critic_forward(post)
        ita = self.config.ita
        rho = self.config.rho
        """imag_value & V_lambda are all from target_critic"""
        reinforce_loss = -torch.mean(act_log_probs.unsqueeze(-1) * (V_lambda - imag_value).detach())
        dynamic_bp_loss = -torch.mean(V_lambda)
        entropy_loss = torch.mean(act_ent)
        actor_loss = rho * reinforce_loss + (1 - rho) * dynamic_bp_loss + ita * entropy_loss
        self.optimizer['actor'].zero_grad()
        actor_loss.backward()
        self.optimizer['actor'].step()

        critic_loss = -torch.mean(value_dist.log_prob(V_lambda.detach()))

        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        self.optimizer['critic'].step()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer['actor'].step()
        if self.scheduler is not None:
            self.scheduler['actor'].step()

        # Logger
        model_lr = self.optimizer['model'].state_dict()['param_groups'][0]['lr']
        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "loss/model_loss": model_loss.item(),
            "loss/obs_loss": obs_loss.item(),
            "loss/rew_loss": rew_loss.item(),
            "loss/noterm_loss": noterm_loss.item(),
            "loss/kl_loss": kl_loss.item(),


            "loss/actor_loss": actor_loss.item(),
            "loss/critic_loss": critic_loss.item(),
            "loss/entropy_loss": entropy_loss.item(),

            # TODO

            "lr/model_lr": model_lr,
            "lr/actor_lr": actor_lr,
            "lr/critic_lr": critic_lr,
        }

        return info
