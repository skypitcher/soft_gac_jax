<h1 align="center">Generative Actor-Critic with Soft Bridge Policies</h1>

<p align="center">
  <b>NeurIPS 2026</b><br>
  Ke He · Le He · Shunpu Tang · Yafei Wang · Lisheng Fan
</p>

<p align="center">
  <a href="https://skypitcher.github.io/soft_gac_home/"><img src="https://img.shields.io/badge/Project%20Page-Visit%20%E2%86%92-0a5cc2?style=for-the-badge" alt="Project page"></a>
  <a href="https://skypitcher.github.io/soft_gac_home/#video"><img src="https://img.shields.io/badge/Video-4%20min-e2701b?style=for-the-badge" alt="Overview video"></a>
  <a href="https://arxiv.org/abs/2605.08733"><img src="https://img.shields.io/badge/arXiv-2605.08733-b31b1b?style=for-the-badge" alt="arXiv"></a>
  <a href="preprint.pdf"><img src="https://img.shields.io/badge/Paper-PDF-555b66?style=for-the-badge" alt="Paper PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0f8a5f?style=for-the-badge" alt="MIT license"></a>
</p>

<!-- To play the overview video inline: edit this file on GitHub, drag softgac_teaser_en_github.mp4 (under 10 MB)
     onto the line below, and GitHub inserts a https://github.com/user-attachments/assets/... link that renders as a player.
     The thumbnail block after it can then be removed. -->

<p align="center">
  <a href="https://skypitcher.github.io/soft_gac_home/#video"><img src="assets/softgac_video_thumbnail.jpg" width="760" alt="SoftGAC overview video: click to watch on the project page"></a><br>
  <sub>▶ Watch the 4-minute overview on the <a href="https://skypitcher.github.io/soft_gac_home/">project page</a> (or download the <a href="https://skypitcher.github.io/soft_gac_home/assets/videos/softgac_overview_en.mp4">MP4</a>).</sub>
</p>

Official **JAX** implementation of **SoftGAC**, a one-step generative actor for maximum-entropy RL. From the outside it is a single network call; inside, a short stochastic **soft bridge** of Gaussian layers makes the soft regularizer an **exact, analytical path KL**.

- **Soft:** principled MaxEnt, endpoint-equivalent to SAC. No entropy estimates or surrogates.
- **Fast:** one network call per action, with about the same number of parameters as the baselines.
- **Better:** +44% average return over the strongest many-step diffusion policy on 8 hard locomotion tasks, at about 1/6 of its latency.

<p align="center">
  <img src="assets/softgac_soft_bridge.png" width="900" alt="The soft bridge actor: one network call whose K Gaussian transition layers (about M/K parameters each) are paired with a fixed, parameter-free maximum-entropy reference process; the per-layer KLs add up to the exact path KL. Below, a many-step actor reuses one network of M parameters N times, while SoftGAC splits the same budget into K layers used once."><br>
  <sub>Open the box: <i>K</i> Gaussian transition layers share the parameter budget of one network, and their closed-form KLs to a max-entropy reference add up to the exact soft regularizer.</sub>
</p>

<p align="center">
  <img src="assets/softgac_compute_return.png" width="560" alt="Average final IQM return versus per-action inference time: SoftGAC is top-left, +44% over DIME at about 1/6 of its latency"><br>
  <sub>Faster does not have to mean weaker.</sub>
</p>

<p align="center">
  <img src="assets/softgac_rollouts.gif" width="640" alt="Trained SoftGAC policies on Humanoid Run, Dog Run, H1 Hurdle and H1 Run"><br>
  <sub>Trained SoftGAC policies: Humanoid Run, Dog Run, H1 Hurdle, H1 Run.</sub>
</p>

## Quick start

```bash
git clone https://github.com/skypitcher/soft_gac_jax.git
cd soft_gac_jax
./setup.sh                 # conda env "soft_gac": TPU JAX + CPU-only PyTorch
conda activate soft_gac
python main.py             # SoftGAC on dm_control/dog-run
```

- **CUDA 12 GPUs:** `REQ_FILE=requirements-gpu.txt ./setup.sh`
- **HumanoidBench** (`h1-*-v0` tasks): `./setup_humanoidbench.sh`, then `conda activate soft_gac_humanoidbench`

PyTorch is only used for Stable-Baselines3 replay buffers.

## Usage

```bash
# SoftGAC with K = 6 bridge layers
python main.py alg=soft_gac env_name=dm_control/dog-run alg.actor.hidden_size=256 alg.actor.num_layers=6
python main.py alg=soft_gac env_name=h1-hurdle-v0      alg.actor.hidden_size=512 alg.actor.num_layers=6

# Baselines: sac (CrossQ-SAC), dime, flowrl, flac, qsm, qvpo
python main.py alg=dime env_name=dm_control/humanoid-run

# 8 seeds in parallel (do not pass seed=...)
bash run_seeds_new.sh alg=soft_gac env_name=dm_control/dog-run
```

Any config value can be overridden as `key=value` (defaults in `configs/main.yaml` and `configs/alg/*.yaml`), for example `tot_time_steps=100000`, `wandb.enabled=true` or `checkpoint.enabled=false`. Runs default to 1M environment steps (3M for `dm_control/humanoid-run` and `h1-hurdle-v0`) and are saved to `results/<env>/<run-label>/<timestamp>_seedN/` with logs, `train.csv`, `eval.csv` and a code snapshot.

To time actor inference for all methods (compilation excluded): `python benchmark_actor_forward.py --batch-size 1`.

<details>
<summary><b>Occasional dead seeds on <code>humanoid-run</code></b></summary>

On TPU with `alg.train.num_particles=1`, we saw two dead seeds (6 and 7) on `dm_control/humanoid-run`; the other 11 tasks were unaffected across 8 seeds. More bridge particles lower the actor-update variance and fix it. This only affects training, not action generation:

```bash
python main.py alg=soft_gac env_name=dm_control/humanoid-run alg.actor.hidden_size=512 alg.actor.num_layers=6 alg.train.num_particles=32
python main.py alg=soft_gac env_name=dm_control/humanoid-run alg.actor.hidden_size=512 alg.actor.num_layers=6 alg.train.num_particles=4 alg.train.particle_scheme=antithetic
```
</details>

## Citation

```bibtex
@inproceedings{he2026softgac,
  title     = {Generative Actor-Critic with Soft Bridge Policies},
  author    = {He, Ke and He, Le and Tang, Shunpu and Wang, Yafei and Fan, Lisheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.08733}
}
```

We thank the Google TPU Research Cloud (TRC) program for providing compute resources. Released under the [MIT License](LICENSE).
