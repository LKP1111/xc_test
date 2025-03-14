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
        self.iterations = 0
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        """(n_envs, seq, batch, ~)"""
        obs_seq_batch = torch.as_tensor(samples['obs'], device=self.device).float()
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
        """act -> obs', rew, noterm"""

        """
        model learning
        model_loss = -obs_dist.log_prob(obs_batch) - rew_dist.log_prob(rew_batch) - noterm_dist.log_prob(noterm_batch) + 
                kl_div(prior, sg(post)) + kl_div(sg(prior), post)
        """
        """(seq - 1, n_envs * batch, ~) * 3, (seq, n_envs * batch, ~) * 2"""
        obs_dist, rew_dist, noterm_dist, prior, post = self.policy.model_forward(
            obs_seq_batch,
            act_seq_batch,
            noterm_seq_batch
        )
        prior_dist = self.policy.representation.RSSM.get_dist(prior)
        post_dist = self.policy.representation.RSSM.get_dist(post)
        prior_dist_detach = self.policy.representation.RSSM.get_dist(
            self.policy.representation.RSSM.rssm_detach(prior))
        post_dist_detach = self.policy.representation.RSSM.get_dist(
            self.policy.representation.RSSM.rssm_detach(post))

        """seq_shift: obs_seq_batch[:-1], rew_seq_batch[1:], noterm_seq_batch[1:]"""
        obs_loss = -torch.mean(obs_dist.log_prob(obs_seq_batch[:-1]))
        rew_loss = -torch.mean(rew_dist.log_prob(rew_seq_batch[:-1]))  # after obs
        noterm_loss = -torch.mean(noterm_dist.log_prob(noterm_seq_batch[:-1]))
        alpha = self.config.kl['kl_balance_scale']
        kl_div = torch.distributions.kl.kl_divergence
        """kl_loss checked"""
        kl_l = torch.mean(kl_div(prior_dist, post_dist_detach))
        kl_r = torch.mean(kl_div(prior_dist_detach, post_dist))
        kl_loss = alpha * kl_l + (1 - alpha) * kl_r
        rew_scale = self.config.loss_scale['reward']
        discount_scale = self.config.loss_scale['discount']
        kl_scale = self.config.loss_scale['kl']
        model_loss = obs_loss + rew_scale * rew_loss + discount_scale * noterm_loss + kl_scale * kl_loss
        self.optimizer['model'].zero_grad()
        model_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.model_parameters, self.grad_clip_norm)
        self.optimizer['model'].step()

        """
        actor_critic learning
        actor_loss = -rho * act_dist.log_prob(act_batch) * sg(V_lambda - target_v_dist.sample()) -
                (1 - rho) * V_lambda + act_dist.entropy()
        critic_loss = -v_dist.log_prob(sg(V_lambda))
        """
        act_log_probs, act_ent, imag_value, V_lambda, value_dist, weights \
            = self.policy.actor_critic_forward(post)
        ita = self.config.ita
        rho = self.config.rho

        """
        imag_value & V_lambda are all from target_critic in dreamer-ref
        imag_value is from critic in dreamer-torch
        """
        """seq_shift: act_log_probs[1:], V_lambda[:-1], imag_value[:-1]"""
        # reinforce_loss = -act_log_probs[:-1].unsqueeze(-1) * (V_lambda[:-1] - imag_value[:-1]).detach()
        reinforce_loss = -act_log_probs[:-1].unsqueeze(-1) * (V_lambda[:-1] - value_dist.mean.detach()).detach()
        dynamic_bp_loss = -V_lambda[:-1]
        """seq_shift: act_ent[1:]"""
        entropy_loss = -act_ent[:-1][:,:,None]
        actor_loss = rho * reinforce_loss + (1 - rho) * dynamic_bp_loss + ita * entropy_loss
        actor_loss = torch.mean(actor_loss * weights)
        self.optimizer['actor'].zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
        self.optimizer['actor'].step()

        """seq_shift: V_lambda[:-1]"""
        critic_loss = -value_dist.log_prob(V_lambda[:-1].detach())
        critic_loss = torch.mean(critic_loss * weights)
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()

        # TODO Logger
        model_lr = self.optimizer['model'].state_dict()['param_groups'][0]['lr']
        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        """
        advantage = lambda_return - target_value
        critic_loss = mse(target_value, value) or -maxlikelihood(target_value, value)
        """
        info = {
            "metrics/critic_value_dist_mean": torch.mean(value_dist.mean),
            "metrics/target_critic_value": torch.mean(imag_value[:-1]),
            "metrics/lambda_return": torch.mean(V_lambda[:-1]),

            "model_loss/model_loss": model_loss.item(),
            "model_loss/obs_loss": obs_loss.item(),
            "model_loss/rew_loss": rew_loss.item(),
            "model_loss/noterm_loss": noterm_loss.item(),
            "model_loss/kl_loss": kl_loss.item(),

            "actor_loss/actor_loss": torch.mean(actor_loss).item(),
            "actor_loss/reinforce_loss": torch.mean(reinforce_loss).item(),
            "actor_loss/dynamic_bp_loss(-V_lambda)": torch.mean(dynamic_bp_loss).item(),
            "actor_loss/entropy_loss": torch.mean(entropy_loss).item(),

            "critic_loss": torch.mean(critic_loss).item(),

            "update_count": self.iterations

            # "lr/model_lr": model_lr,
            # "lr/actor_lr": actor_lr,
            # "lr/critic_lr": critic_lr,
        }

        return info
