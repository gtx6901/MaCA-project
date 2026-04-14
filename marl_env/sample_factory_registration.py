"""Registration helpers for Sample Factory integration."""

from __future__ import annotations

from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.utils.utils import str2bool

from marl_env.sample_factory_env import SampleFactoryMaCAEnv
from marl_env.sample_factory_model import MaCAEncoder


_ENV_PREFIX = "maca_"
_ENCODER_NAME = "maca_simple"
_REGISTERED = False


def make_maca_env(full_env_name, cfg=None, env_config=None):
    return SampleFactoryMaCAEnv(full_env_name, cfg=cfg, env_config=env_config)


def add_maca_env_args(env, parser):
    del env
    p = parser
    p.add_argument("--maca_map_path", default="maps/1000_1000_fighter10v10.map", type=str, help="MaCA map path")
    p.add_argument("--maca_red_obs_ind", default="simple", type=str, help="Red observation constructor")
    p.add_argument("--maca_opponent", default="fix_rule", type=str, help="Opponent agent package name")
    p.add_argument("--maca_max_step", default=1000, type=int, help="Episode horizon for MaCA")
    p.add_argument("--maca_render", default=False, type=str2bool, help="Enable the original pygame renderer")
    p.add_argument("--maca_random_pos", default=False, type=str2bool, help="Randomize initial side positions")
    p.add_argument(
        "--maca_extended_observation",
        default=False,
        type=str2bool,
        help="Append explicit contact/attack-envelope indicators to fighter measurements",
    )
    p.add_argument(
        "--maca_decoupled_action_heads",
        default=False,
        type=str2bool,
        help="Use tuple action heads: course head + attack head",
    )
    p.add_argument(
        "--maca_adaptive_support_policy",
        default=False,
        type=str2bool,
        help="Use radar/jammer receive history to choose radar and disturb frequencies",
    )
    p.add_argument(
        "--maca_support_search_hold",
        default=6,
        type=int,
        help="How many steps to hold a sampled radar point when no receive cue is available",
    )
    p.add_argument(
        "--maca_radar_tracking_observation",
        default=False,
        type=str2bool,
        help="Append radar receive and target-track memory features to fighter measurements",
    )
    p.add_argument(
        "--maca_track_memory_steps",
        default=12,
        type=int,
        help="How long to keep last seen target track features after contact is lost",
    )
    p.add_argument(
        "--maca_semantic_screen_observation",
        default=False,
        type=str2bool,
        help="Replace raw id/type image planes with semantic occupancy and track-memory planes",
    )
    p.add_argument(
        "--maca_screen_track_memory_steps",
        default=12,
        type=int,
        help="How long to keep image-space enemy track memory after current detections disappear",
    )


def maca_override_defaults(env, parser):
    del env
    parser.set_defaults(
        encoder_type="conv",
        encoder_custom=_ENCODER_NAME,
        encoder_subtype="convnet_simple",
        hidden_size=256,
        encoder_extra_fc_layers=1,
        actor_critic_share_weights=True,
        use_rnn=True,
        rnn_type="lstm",
        rnn_num_layers=1,
        rollout=64,
        recurrence=64,
        batch_size=5120,
        num_workers=8,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=50000000,
        save_every_sec=300,
        keep_checkpoints=20,
        experiment_summaries_interval=30,
        stats_avg=50,
        learning_rate=1e-4,
        gamma=0.999,
        reward_scale=0.005,
        reward_clip=50.0,
        ppo_epochs=4,
        max_grad_norm=0.5,
        max_policy_lag=15,
        exploration_loss_coeff=0.02,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=1,
    )


def register_maca_components():
    global _REGISTERED
    if _REGISTERED:
        return

    registry = global_env_registry()
    if _ENV_PREFIX not in registry.registry:
        registry.register_env(
            env_name_prefix=_ENV_PREFIX,
            make_env_func=make_maca_env,
            add_extra_params_func=add_maca_env_args,
            override_default_params_func=maca_override_defaults,
        )

    register_custom_encoder(_ENCODER_NAME, MaCAEncoder)
    _REGISTERED = True
