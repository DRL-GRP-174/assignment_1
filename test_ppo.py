"""Sanity checks we wrote before trusting the training curves.
run: python test_ppo.py"""
import torch

from ppo import compute_gae, ppo_loss


def test_gae_lambda1_equals_mc_advantage():
    # with lambda = 1 and no terminal, GAE must equal (discounted MC return) - V(s)
    T, g = 5, 0.9
    r = torch.rand(T, 1)
    v = torch.rand(T, 1)
    last_v = torch.rand(1)
    adv, _ = compute_gae(r, v, torch.zeros(T, 1), last_v, torch.zeros(1), g, 1.0)
    ret, mc = last_v.clone(), torch.zeros(T, 1)
    for t in reversed(range(T)):
        ret = r[t] + g * ret
        mc[t] = ret
    assert torch.allclose(adv, mc - v, atol=1e-5)


def test_gae_lambda0_is_td_error():
    r, v = torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [0.2]])
    adv, _ = compute_gae(r, v, torch.zeros(2, 1), torch.tensor([0.3]), torch.zeros(1), 0.9, 0.0)
    assert abs(adv[0].item() - (1.0 + 0.9 * 0.2 - 0.5)) < 1e-6


def test_clip_kills_gradient_outside_trust_zone():
    # A > 0 and ratio already 1.5 (> 1+eps): clipped objective should give zero gradient
    old = torch.log(torch.tensor([0.2]))
    new = torch.log(torch.tensor([0.3])).requires_grad_()   # r = 1.5
    loss, _ = ppo_loss(new, old, torch.tensor([1.0]), eps=0.2)
    loss.backward()
    assert new.grad.abs().item() == 0.0


def test_clip_keeps_gradient_when_update_hurts():
    # A < 0 and ratio 1.5: moving further is bad, so min() keeps the unclipped term -> non-zero grad
    old = torch.log(torch.tensor([0.2]))
    new = torch.log(torch.tensor([0.3])).requires_grad_()
    loss, _ = ppo_loss(new, old, torch.tensor([-1.0]), eps=0.2)
    loss.backward()
    assert new.grad.abs().item() > 0.0


def test_no_clip_matches_plain_surrogate():
    old = torch.log(torch.tensor([0.2, 0.5]))
    new = torch.log(torch.tensor([0.3, 0.4]))
    adv = torch.tensor([1.0, -2.0])
    loss, ratio = ppo_loss(new, old, adv, eps=0.0)
    assert torch.allclose(loss, -(ratio * adv).mean())


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED  {t.__name__}")
    print(f"\n{len(tests)} passed")
