from contextlib import contextmanager
import threading
from typing import Any

import torch

from mace.tools.scatter import scatter_sum


_TRACE_PATCH_LOCK = threading.Lock()


def assemble_edge_forces(
    edge_gradient: torch.Tensor,
    edge_index: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    sender, receiver = edge_index[0], edge_index[1]
    forces = torch.zeros(
        (num_atoms, 3),
        dtype=edge_gradient.dtype,
        device=edge_gradient.device,
    )
    forces = forces.index_add(0, sender, edge_gradient)
    forces = forces.index_add(0, receiver, -edge_gradient)
    return forces


def assemble_edge_virials_stress(
    edge_gradient: torch.Tensor,
    vectors: torch.Tensor,
    edge_index: torch.Tensor,
    node_batch: torch.Tensor,
    cell: torch.Tensor,
    num_graphs: int,
    compute_stress: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    edge_graph = node_batch[edge_index[0]]
    displacement_gradient = scatter_sum(
        torch.einsum("ei,ej->eij", vectors, edge_gradient),
        edge_graph,
        dim=0,
        dim_size=num_graphs,
    )
    displacement_gradient = 0.5 * (
        displacement_gradient + displacement_gradient.transpose(-1, -2)
    )
    virials = -displacement_gradient
    if not compute_stress:
        return virials, None
    volume = torch.linalg.det(cell.view(-1, 3, 3)).abs().view(-1, 1, 1)
    stress = displacement_gradient / volume
    stress = torch.where(
        torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
    )
    return virials, stress


@contextmanager
def trace_edge_derivative_outputs():
    from mace.modules import models

    with _TRACE_PATCH_LOCK:
        original_prepare = models.prepare_graph
        original_outputs = models.get_outputs
        trace_state: dict[str, Any] = {}

        def prepare_graph(data, **kwargs):
            context = original_prepare(data, **kwargs)
            vectors = context.vectors.detach().requires_grad_(True)
            trace_state["edge_index"] = data["edge_index"]
            trace_state["node_batch"] = data["batch"]
            trace_state["num_graphs"] = context.num_graphs
            return context._replace(
                vectors=vectors,
                lengths=torch.linalg.vector_norm(vectors, dim=1, keepdim=True),
            )

        def get_outputs(
            energy,
            positions,
            cell,
            displacement,
            vectors=None,
            magmoms=None,
            training=False,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_magforces=False,
        ):
            del displacement
            if (
                vectors is None
                or magmoms is not None
                or compute_hessian
                or compute_edge_forces
                or compute_magforces
            ):
                raise RuntimeError(
                    "edge derivative training supports only energy/force/virial/stress"
                )
            edge_gradient = torch.autograd.grad(
                energy,
                vectors,
                grad_outputs=torch.ones_like(energy),
                retain_graph=training,
                create_graph=training,
                allow_unused=False,
            )[0]
            forces = (
                assemble_edge_forces(
                    edge_gradient,
                    trace_state["edge_index"],
                    positions.shape[0],
                )
                if compute_force
                else None
            )
            virials = stress = None
            if compute_virials or compute_stress:
                virials, stress = assemble_edge_virials_stress(
                    edge_gradient,
                    vectors,
                    trace_state["edge_index"],
                    trace_state["node_batch"],
                    cell,
                    trace_state["num_graphs"],
                    compute_stress,
                )
            return forces, virials, stress, None, None, None

        models.prepare_graph = prepare_graph
        models.get_outputs = get_outputs
        try:
            yield
        finally:
            models.prepare_graph = original_prepare
            models.get_outputs = original_outputs
