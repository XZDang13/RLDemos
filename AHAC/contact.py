"""Contact forces and body accelerations of the last physics substep of a Rewarped Featherstone task."""

import torch


class ContactProbe:
    """Callable ``probe(env) -> (contact_forces, accelerations)``, both ``[N, bodies, 3]``.

    Rewarped does not report contact forces. Warp's Featherstone integrator
    accumulates rigid contact wrenches into ``state_in.body_f`` and writes the
    spatial body accelerations ``body_a_s`` to ``state_out`` (with gradients) or
    to the integrator itself, the same quantities the reference AHAC reads from
    dflex. The probe wraps ``integrator.simulate`` and keeps references to the
    states of the latest call; with CUDA graph capture that call happens once
    and the references point at the buffers every replay writes to, so create
    the probe before the first ``env.step``.
    """

    def __init__(self, env):
        integrator = env.integrator
        if not hasattr(integrator, "allocate_state_aux_vars"):
            raise ValueError("ContactProbe requires a task that uses the Featherstone integrator.")
        self.num_envs = env.num_envs
        self.state_in = None
        self.state_aug = None
        simulate = integrator.simulate

        def recording_simulate(model, state_in, state_out, dt, control=None):
            simulate(model, state_in, state_out, dt, control=control)
            self.state_in = state_in
            self.state_aug = state_out if state_in.requires_grad else integrator

        integrator.simulate = recording_simulate

    @torch.no_grad()
    def __call__(self, env):
        if self.state_in is None:
            raise RuntimeError("ContactProbe has not seen a physics step yet.")
        import warp as wp

        # Warp spatial vectors are (angular, linear).
        contact_forces = wp.to_torch(self.state_in.body_f).view(self.num_envs, -1, 6)[..., 3:].clone()
        accelerations = wp.to_torch(self.state_aug.body_a_s).view(self.num_envs, -1, 6)[..., 3:].clone()
        return contact_forces, accelerations
